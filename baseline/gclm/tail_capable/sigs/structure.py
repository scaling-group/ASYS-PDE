"""SIGS-derived stored-atom sampling and latent proposals, with explicit domains.

This is an adaptation of the released sampler, not its hardcoded PDE driver.
No saved corpus/vector is rewritten. A stored expression and a decoded mean
are different proposals; only their own parsed expressions are evaluated.
"""

import ast
import collections
import math
import pickle

import numpy as np
import torch
from expression import Expression, Profile
from mpmath import iv


def canonical(text):
    """Alpha-rename exactly one coordinate; never convert a multivariate atom."""
    tree = ast.parse(text.replace("^", "**"), mode="eval")
    coordinates = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & {"x", "y", "t"}
    if len(coordinates) > 1:
        raise ValueError("Multivariate atom")

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in coordinates:
                return ast.copy_location(ast.Name(id="x", ctx=ast.Load()), node)
            return node

    text = ast.unparse(ast.fix_missing_locations(Rename().visit(tree)))
    Expression(text)  # Reject every other unrecognized token/operator.
    return text


def domain(expression, subdivisions=32):
    """Conservative interval enclosure on [0,1], including finite endpoints.

    Passing is sufficient, not necessary. Expressions with removable singular
    subexpressions may fail. This is a rejection rule, never expression repair.
    Interval work is at 30 decimal digits; float parameters enter exactly.
    """
    iv.dps = 30
    coefficients = expression.coefficients.detach().tolist()

    def enclose(node, coordinate):
        kind = node[0]
        if kind == "constant":
            return iv.mpf(node[1])
        if kind == "parameter":
            return iv.mpf(coefficients[node[1]])
        if kind == "name":
            # Bounds cover the actual float constants used by the evaluator.
            return coordinate if node[1] == "x" else iv.mpf(getattr(math, node[1]))
        if kind == "sign":
            return node[1] * enclose(node[2], coordinate)
        if kind == "function":
            value = enclose(node[2], coordinate)
            if node[1] == "log" and float(value.a) <= 0:
                raise ValueError("Log argument not strictly positive")
            if node[1] == "sqrt" and float(value.a) < 0:
                raise ValueError("Square-root argument may be negative")
            if node[1] == "tanh":
                return 2 / (1 + iv.exp(-2 * value)) - 1
            return getattr(iv, node[1])(value)
        left, right = enclose(node[1], coordinate), enclose(node[2], coordinate)
        if kind == "Add":
            return left + right
        if kind == "Sub":
            return left - right
        if kind == "Mult":
            return left * right
        if kind == "Div":
            if float(right.a) <= 0 <= float(right.b):
                raise ValueError("Denominator enclosure contains zero")
            return left / right
        lower = float(right.a)
        # Do not infer a point integer from rounded float endpoints: a narrow
        # interval around 3 may round to 3 at both ends but include nonintegers.
        integer = (
            right.a == right.b
            and math.isfinite(lower)
            and lower.is_integer()
            and right.a == iv.mpf(int(lower))
        )
        if float(left.a) < 0 and not integer:
            raise ValueError("Noninteger power may have negative base")
        if float(left.a) <= 0 <= float(left.b) and lower <= 0 and not (integer and lower == 0):
            raise ValueError("Nonpositive power may have zero base")
        return iv.power(left, right)

    def interval(left, right):
        value = enclose(expression.tree, iv.mpf([left, right]))
        low, high = float(value.a), float(value.b)
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError("Nonfinite enclosure")
        return low, high

    try:
        low, high = interval(0.0, 1.0)
        return dict(admissible=True, lower=low, upper=high, intervals=1)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        pass
    try:
        ranges = [interval(i / subdivisions, (i + 1) / subdivisions) for i in range(subdivisions)]
        return dict(
            admissible=True,
            lower=min(v[0] for v in ranges),
            upper=max(v[1] for v in ranges),
            intervals=subdivisions,
        )
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return dict(admissible=False, failure=str(exc), intervals=subdivisions)


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("collections", "defaultdict"): collections.defaultdict,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unexpected type {module}.{name}")
        return allowed[module, name]


def library(upstream):
    """Return accepted original atoms plus every exclusion reason; no fitting."""
    with (upstream / "assets/clusters.pkl").open("rb") as stream:
        raw = RestrictedUnpickler(stream).load()["SPATIAL_1D"]
    accepted, rejected = [], []
    for index, (text, vector) in enumerate(zip(raw["expressions"], raw["vectors"], strict=True)):
        row = dict(index=index, original=text)
        try:
            interpreted = canonical(text)
            expression = Expression(interpreted)
            certificate = domain(expression)
            if not certificate["admissible"]:
                raise ValueError(certificate["failure"])
            has_x = any(
                isinstance(n, ast.Name) and n.id == "x"
                for n in ast.walk(ast.parse(interpreted, mode="eval"))
            )
            if not has_x:
                raise ValueError("Constant stored atom contributes no centered correction")
            row.update(
                expression=interpreted, vector=np.asarray(vector, dtype=float), domain=certificate
            )
            accepted.append(row)
        except (ValueError, SyntaxError, OverflowError) as exc:
            row["failure"] = str(exc)
            rejected.append(row)
    return accepted, rejected


def assemble(atoms, amplitude=0.1):
    return " + ".join(f"{amplitude!r}*({canonical(atom)})" for atom in atoms)


def formula(model):
    expression = model.expression
    return dict(
        expression=expression.original,
        parameterized=expression.parameterized,
        coefficients=expression.coefficients.detach().tolist(),
        beta=float(model.beta()),
        scalar_count=sum(p.numel() for p in model.parameters()),
        coordinate="v=(1+x^2)^(-1/(2*beta))",
        correction="E(v)-E(1)",
        raw_profile="x*exp(-(1+1/beta)/2*log(1+x^2)+E(v)-E(1))",
    )


class Proposals:
    def __init__(self, atoms, slots, seed, clusters=5):
        self.atoms, self.slots, self.clusters = atoms, slots, clusters
        self.rng = np.random.default_rng(seed)

    def partition(self, indices):
        from sklearn.cluster import KMeans

        indices = np.asarray(indices, dtype=int)
        vectors = np.array([self.atoms[i]["vector"] for i in indices])
        count = min(self.clusters, len(np.unique(vectors, axis=0)))
        if count == 0:
            raise ValueError("No atoms to partition")
        labels = KMeans(n_clusters=count, n_init=10, random_state=42).fit_predict(vectors)
        return [indices[labels == k] for k in range(count)]

    def broad(self):
        groups = self.partition(np.arange(len(self.atoms)))
        return [groups for _ in range(self.slots)]

    def refine(self, memberships):
        return [self.partition(group) for group in memberships]

    def sample(self, groups, count):
        """Balance cluster tuples, count every draw, and never retry an invalid fit."""
        import itertools

        combinations = list(itertools.product(*(range(len(g)) for g in groups)))
        self.rng.shuffle(combinations)
        for ordinal in range(count):
            combo = combinations[ordinal % len(combinations)]
            memberships = [groups[slot][cluster] for slot, cluster in enumerate(combo)]
            indices = [int(self.rng.choice(group)) for group in memberships]
            yield dict(
                kind="stored",
                indices=indices,
                memberships=memberships,
                atoms=[self.atoms[i]["expression"] for i in indices],
                original_indices=[self.atoms[i]["index"] for i in indices],
                latent=np.concatenate([self.atoms[i]["vector"] for i in indices]),
            )


class Decoder:
    def __init__(self, upstream):
        from sigs.model import GrammarVAE

        checkpoint = torch.load(
            upstream / "assets/model.ckpt", map_location="cpu", weights_only=True
        )
        # Model construction must remain float32 even when the fitter uses float64.
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            self.model = GrammarVAE(checkpoint["hyper_parameters"]["config"])
        finally:
            torch.set_default_dtype(previous)
        state = checkpoint["state_dict"]
        if sorted(k for k in state if not k.startswith("model.")) != ["directions.U"]:
            raise ValueError("Unexpected checkpoint state")
        self.model.load_state_dict(
            {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")},
            strict=True,
        )
        self.model.eval()

    def __call__(self, flat, slots):
        from sigs.utils import ExpressionUtils

        atoms = []
        with torch.no_grad():
            for vector in np.split(np.asarray(flat), slots):
                z = torch.tensor(vector, dtype=torch.float32).reshape(1, 1, -1)
                logits = self.model.decoder(z)[0, 0]
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Nonfinite decoder logits")
                atoms.append(
                    ExpressionUtils.construct_expression(logits, self.model.max_length, "cpu")
                )
        return atoms


def candidate(proposal, beta=1.0, amplitude=0.1):
    model = Profile(assemble(proposal["atoms"], amplitude), beta)
    certificate = domain(model.expression)
    if not certificate["admissible"]:
        raise ValueError(certificate["failure"])
    return model
