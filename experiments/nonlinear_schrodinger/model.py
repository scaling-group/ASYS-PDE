"""Direct source predictor and nonlinear-defect response with shared weights.

Core/radiation and quadrature adapt solver_f741da21f5d3. Four dressed
perturbation orders adapt solver_6eee4ec7505f. New seeds recycle their exact
nonlinear integral defect using the linearization about that source predictor.
All learned deviations start at zero; construction is a bounded operator formula.
Fixed predictor controls translate the original eight conditioned coordinates.
"""
import math
import numpy as np
import torch
from problem import core, initial, T_MAX, TRAIN_WIDTHS


def line_field(x, t, width):
    baseline = torch.cat((initial(x, width), torch.zeros_like(x)), dim=-1)
    uv = baseline + (core(x, t) - core(x, torch.zeros_like(t)))
    return torch.complex(uv[..., 0], uv[..., 1])


def response(omega, frequency, t):
    """Zero-IC solution of i H' - omega H = exp(i frequency t)."""
    return (-1j*t * torch.exp(0.5j*(frequency-omega)*t)
            * torch.sinc((omega+frequency)*t/(2*math.pi)))


def cubic_difference(c, d):
    """N(c+d)-N(c), expanded to avoid subtracting nearly equal cubics."""
    return (2*c.abs().square()*d+c.square()*d.conj()
            +2*c*d.abs().square()+c.conj()*d.square()+d.abs().square()*d)


def quadratic(c, u, v):
    return 2*c*(u*v.conj()).real+c.conj()*u*v


def cubic(u, v, w):
    return (u*v*w.conj()+u*w*v.conj()+v*w*u.conj())/3


def chebyshev(z, degree):
    values = [torch.ones_like(z), z]
    for _ in range(2, degree+1):
        values.append(2*z*values[-1]-values[-2])
    return torch.stack(values[:degree+1], dim=-1)


class WidthBasis(torch.autograd.Function):
    """Bound reverse-mode memory for source setup using a forward AD tangent.

    The first width derivative is the tangent of the actual finite quadrature.
    Coordinate x/t differentiation never records the construction graph.
    """
    @staticmethod
    def forward(ctx, width, model):
        ctx.model = model
        ctx.save_for_backward(width)
        return model.basis(width.detach())

    @staticmethod
    def backward(ctx, gradient):
        width, = ctx.saved_tensors
        tangent = WidthDerivative.apply(width, ctx.model, 1)
        return (gradient.conj()*tangent).real.sum(), None


class WidthDerivative(torch.autograd.Function):
    """Forward AD derivatives, with recursive support for higher derivatives."""
    @staticmethod
    def forward(ctx, width, model, order):
        ctx.model, ctx.order = model, order
        ctx.save_for_backward(width)
        key = (float(width), str(width.device), width.dtype, order)
        if key not in model._tangent_cache:
            function = model._make_basis
            for _ in range(order):
                previous = function
                function = lambda w, previous=previous: torch.func.jvp(
                    previous, (w,), (torch.ones_like(w),))[1]
            with torch.no_grad():
                tangent = function(width.detach())
            if len(model._tangent_cache) >= model.cache_widths:
                model._tangent_cache.pop(next(iter(model._tangent_cache)))
            model._tangent_cache[key] = tangent
        return model._tangent_cache[key]

    @staticmethod
    def backward(ctx, gradient):
        width, = ctx.saved_tensors
        tangent = WidthDerivative.apply(width, ctx.model, ctx.order+1)
        return (gradient.conj()*tangent).real.sum(), None, None


class Candidate(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.fixed_predictor_order = config.get('fixed_predictor_order', 1)
        if self.fixed_predictor_order not in (0, 1, 2):
            raise ValueError('fixed_predictor_order must be 0, 1, or 2')
        self.radiation_modes = int(config.get('radiation_modes', 64))
        self.feedback_modes = int(config.get('feedback_modes', 24))
        self.chain_size = int(config.get('source_chain_size', 4))
        self.krylov_terms = 2*self.chain_size
        self.dressing_terms = int(config.get('dressing_terms', 40))
        self.time_degree = int(config.get('time_degree', 256))
        self.space_nodes = int(config.get('basis_space_nodes', 256))
        self.integral_nodes = int(config.get('integral_nodes', 320))
        self.integration_block = int(config.get('integration_block', 8))
        self.cache_widths = int(config.get('cache_widths', 8))
        self.query_block = int(config.get('query_block', 256))
        if min(self.chain_size, self.cache_widths, self.query_block) < 1 or self.dressing_terms < 0:
            raise ValueError('Invalid source-chain, dressing, or cache size')
        self.coefficients = torch.nn.Parameter(torch.zeros(self.krylov_terms, dtype=torch.float64))
        self.register_buffer('radiation_n', torch.arange(self.radiation_modes+1, dtype=torch.float64))
        self.register_buffer('feedback_n', torch.arange(self.feedback_modes+1, dtype=torch.float64))
        self.register_buffer('source_m', torch.arange(-3, 5, dtype=torch.float64))
        degree = self.time_degree
        z = -np.cos(np.pi*np.arange(degree+1)/degree)
        v = np.cos(np.arccos(z)[:, None]*np.arange(degree+1)[None, :])
        endpoints = np.ones(degree+1)
        endpoints[[0, -1]] = 0.5
        transform = (2/degree)*v.T*endpoints[:, None]*endpoints[None, :]
        integrals = np.zeros(degree+1)
        integrals[::2] = 1/(1-np.arange(0, degree+1, 2)**2)
        self.register_buffer('time_u', torch.as_tensor((z+1)/2))
        self.register_buffer('time_transform', torch.as_tensor(transform))
        self.register_buffer('time_weights', torch.as_tensor(integrals@transform))
        self.register_buffer('barycentric_weights', torch.as_tensor((-1.0)**np.arange(degree+1)*endpoints))
        self.register_buffer('basis_x', (2*torch.arange(self.space_nodes, dtype=torch.float64)+1)/self.space_nodes-1)
        q, w = np.polynomial.legendre.leggauss(self.integral_nodes)
        self.register_buffer('integral_u', torch.as_tensor((q+1)/2))
        self.register_buffer('integral_weights', torch.as_tensor(w/2))
        self.register_buffer('recurrence', torch.zeros(self.krylov_terms, self.krylov_terms, dtype=torch.float64))
        self.register_buffer('seed_norms', torch.ones(2, dtype=torch.float64))
        self.register_buffer('preconditioner', torch.eye(self.krylov_terms, dtype=torch.float64))
        self._basis_cache = {}
        self._tangent_cache = {}
        self._training_cache = None
        # Joint Arnoldi and Gram conditioning sample only analytical sources.
        # No field fitting, coefficient update, or trajectory solve occurs here.
        with torch.no_grad():
            self._initialize_basis(config)
            # Translate only after the unchanged joint recurrence and original
            # Gram conditioning are complete. No construction cache uses order.
            alpha = torch.zeros_like(self.coefficients)
            if self.fixed_predictor_order >= 1:
                alpha[0] = self.seed_norms[0]*self.recurrence[0, 0]
            if self.fixed_predictor_order >= 2:
                # The second seed includes its projections onto the first
                # chain as well as its own diagonal recurrence entry.
                alpha = alpha+self.seed_norms[1]*self.recurrence[:, self.chain_size]
            self.register_buffer('fixed_offset', torch.linalg.solve(self.preconditioner, alpha))

    def total_coefficients(self):
        """Original conditioned coordinates: fixed source plus deviations."""
        return self.fixed_offset+self.coefficients

    def _apply(self, function, recurse=True):
        # Ordinary attributes are regenerated from source after a device/dtype
        # move; they are deliberately excluded from the exported state.
        before = (self.coefficients.device, self.coefficients.dtype)
        result = super()._apply(function, recurse=recurse)
        if before != (self.coefficients.device, self.coefficients.dtype):
            self._basis_cache = {}
            self._tangent_cache = {}
            self._training_cache = None
        return result

    def source_coefficients(self, width):
        """Eight analytical harmonics of the supplied core_x(L,t)."""
        width = width[..., None]
        a = torch.cosh(4*width)+4*torch.cosh(2*width)
        ax = 4*torch.sinh(4*width)+8*torch.sinh(2*width)
        d = torch.sqrt(a.square()-9)
        r = -3/(a+d)
        A, Ax = torch.cosh(3*width), 3*torch.sinh(3*width)
        B, Bx = torch.cosh(width), torch.sinh(width)
        def terms(m):
            rm = r.pow(m.abs().to(torch.int64))
            return rm/d, rm*(a/d.pow(3)+m.abs()/d.square())
        g, h = terms(self.source_m)
        gm, hm = terms(self.source_m-1)
        return 4*(Ax*g+3*Bx*gm-ax*(A*h+3*B*hm))

    def radiation_amplitudes(self, t, width):
        if not torch.is_grad_enabled() and t.ndim == 1 and len(t) > 256:
            return torch.cat([self.radiation_amplitudes(tt, ll)
                              for tt, ll in zip(t.split(256), width.split(256))])
        omega = 0.5*(math.pi*self.radiation_n/width[..., None]).square()
        kernels = response(omega[..., :, None], 0.5+4*self.source_m, t[..., None, None])
        sign = torch.where(self.radiation_n.to(torch.int64)%2 == 0, 1.0, -1.0)
        source = self.source_coefficients(width)/(2*width[..., None])
        return (kernels*source[..., None, :]).sum(dim=-1)*sign

    def _geometry(self, width):
        t, x = self.time_u*T_MAX, self.basis_x*width
        tt, xx = torch.meshgrid(t, x, indexing='ij')
        c = line_field(xx.reshape(-1, 1), tt.reshape(-1, 1), width.expand(xx.numel(), 1)).reshape_as(xx)
        cosine = torch.cos(math.pi*self.basis_x[:, None]*self.feedback_n)
        multiplicity = torch.where(self.feedback_n == 0, 1.0, 2.0)
        reconstruction = (cosine*multiplicity).to(c.dtype)
        cosr = torch.cos(math.pi*self.basis_x[:, None]*self.radiation_n)
        mr = torch.where(self.radiation_n == 0, 1.0, 2.0)
        b = self.radiation_amplitudes(t, width.expand_as(t))@(cosr*mr).to(c.dtype).T
        analysis = cosine.to(c.dtype)/self.space_nodes
        source0 = self.source_coefficients(width).sum()/(2*width)
        sign = torch.where(self.radiation_n.to(torch.int64)%2 == 0, 1.0, -1.0)
        seam = (sign*source0).to(c.dtype)@(cosr*mr).to(c.dtype).T
        # F=N(C+B)-N(C): F'(0)=-i*C(0)^2*S(x,0); (-H F)''=i F'.
        endpoint = (0.5*T_MAX*T_MAX*c[0].real.square()*seam)@analysis
        geometry = dict(c=c, b=b, analysis=analysis, reconstruction=reconstruction,
                        endpoint=endpoint, width=width)
        self._set_background(geometry, c+b)
        return geometry

    @staticmethod
    def _set_background(geometry, background):
        # Fixed Galerkin interaction tensors save repeated physical-space
        # reconstruction/projection in the many explicit operator applications.
        geometry['background'] = background
        analysis, reconstruction = geometry['analysis'], geometry['reconstruction']
        geometry['density'] = torch.einsum('tx,xn,xm->tnm',
            (2*background.abs().square()).to(background.dtype), analysis, reconstruction)
        geometry['pair'] = torch.einsum('tx,xn,xm->tnm',
            background.square(), analysis, reconstruction)

    def _duhamel_matrix(self, width):
        """Explicit GL integration of cardinal source polynomials, not a solve.

        H[n,target,source] maps source samples to the zero-IC response.
        Processing eight target times bounds quadrature temporaries.
        """
        omega = 0.5*(math.pi*self.feedback_n/width).square()
        pieces = []
        for u in self.time_u.split(self.integration_block):
            s = u[:, None]*self.integral_u[None, :]
            delta = s[..., None]-self.time_u
            close = delta.abs() < 4*torch.finfo(delta.dtype).eps
            ratio = self.barycentric_weights/torch.where(close, torch.ones_like(delta), delta)
            cardinal = ratio/ratio.sum(dim=-1, keepdim=True)
            cardinal = torch.where(close.any(dim=-1, keepdim=True), close.to(delta.dtype), cardinal)
            phase = torch.exp(-1j*omega[None, :, None]*(u[:, None, None]-s[:, None, :])*T_MAX)
            integral = torch.einsum('bnq,bqj->nbj', phase*self.integral_weights[None, None, :], cardinal.to(phase.dtype))
            pieces.append(-1j*T_MAX*u[None, :, None]*integral)
        return torch.cat(pieces, dim=1)

    def _response_samples(self, source, endpoint, h):
        d = torch.einsum('ntj,jn->tn', h, source)
        safe_u = torch.where(self.time_u == 0, torch.ones_like(self.time_u), self.time_u)
        scaled = d/safe_u[:, None].square()
        return torch.cat((endpoint[None], scaled[1:]), dim=0)

    def _linear_source(self, d, geometry):
        return -(geometry['density']@d[..., None]
                 +geometry['pair']@d.conj()[..., None])[..., 0]

    def _operator(self, scaled, geometry, h):
        return self._response_samples(self._linear_source(
            scaled*self.time_u[:, None].square(), geometry),
            torch.zeros_like(geometry['endpoint']), h)

    def _dress(self, source, geometry, h):
        # Explicit I+A+...+A^J; J is fixed, with no convergence loop or solve.
        value, term = source, source
        for _ in range(self.dressing_terms):
            term = self._operator(term, geometry, h)
            value = value+term
        return value

    def _seeds(self, geometry, h):
        c, b = geometry['c'], geometry['b']
        u2 = self.time_u[:, None].square()
        def field(v):
            return (v*u2)@geometry['reconstruction'].T
        def integrate(f, endpoint=None):
            source = f@geometry['analysis']
            source = torch.cat((torch.zeros_like(source[:1]), source[1:]))
            return self._response_samples(source,
                torch.zeros_like(geometry['endpoint']) if endpoint is None else endpoint, h)
        background = c+b
        first = self._dress(integrate(-cubic_difference(c, b), geometry['endpoint']), geometry, h)
        u = field(first)
        second = self._dress(integrate(-quadratic(background, u, u)), geometry, h)
        v = field(second)
        third = self._dress(integrate(-(2*quadratic(background, u, v)+cubic(u, u, u))), geometry, h)
        w = field(third)
        fourth = self._dress(integrate(-(2*quadratic(background, u, w)
            +quadratic(background, v, v)+3*cubic(u, u, v))), geometry, h)
        predictor = first+second+third+fourth
        p = field(predictor)
        # Defect of the complete nonlinear integral equation, not its series.
        defect = integrate(-cubic_difference(c, b+p), geometry['endpoint'])-predictor
        self._set_background(geometry, background+p)
        correction = self._dress(defect, geometry, h)
        return predictor, correction

    def _inner(self, a, b):
        mult = torch.where(self.feedback_n == 0, 1.0, 2.0)
        widths = a.real.new_tensor(TRAIN_WIDTHS)
        scale = torch.exp(-widths)/widths
        weights = self.time_weights[None, :, None]*mult[None, None, :]/scale[:, None, None].square()
        return ((a.conj()*b).real*weights).sum()/len(TRAIN_WIDTHS)

    def _initialize_basis(self, config):
        geometries = [self._geometry(self.time_u.new_tensor(w)) for w in TRAIN_WIDTHS]
        matrices = [self._duhamel_matrix(g['width']) for g in geometries]
        seeds = [self._seeds(g, h) for g, h in zip(geometries, matrices)]
        u2 = self.time_u[None, :, None].square()
        for k in range(2):
            seed = torch.stack([s[k] for s in seeds])
            self.seed_norms[k] = self._inner(seed*u2, seed*u2).sqrt()
        fields = []
        for j in range(self.krylov_terms):
            value = (torch.stack([s[j//self.chain_size] for s in seeds])/self.seed_norms[j//self.chain_size]
                     if j%self.chain_size == 0 else torch.stack([
                         self._operator(fields[-1][w], g, h)
                         for w, (g, h) in enumerate(zip(geometries, matrices))]))
            # Joint two-pass real modified Gram-Schmidt. This is basis setup.
            for _ in range(2):
                for i in range(j):
                    hij = self._inner(fields[i]*u2, value*u2)
                    self.recurrence[i, j] += hij
                    value = value-hij*fields[i]
            norm = self._inner(value*u2, value*u2).sqrt()
            if not torch.isfinite(norm) or norm < 1e-12:
                raise RuntimeError('Degenerate source-defect basis')
            self.recurrence[j, j] = norm
            fields.append(value/norm)
        del fields
        # Reconstruct from the same seeds/geometry; avoid repeating expensive
        # source dressing while keeping the transfer recurrence identical.
        for w, g, h, source in zip(TRAIN_WIDTHS, geometries, matrices, seeds):
            poly = self._basis_from_seeds(source, g, h)
            self._basis_cache[(w, str(poly.device), poly.dtype)] = poly
        del matrices, geometries, seeds
        from loss import make_cache, residual_basis
        cache = make_cache(self, config)
        basis = residual_basis(self, cache)
        gram = torch.einsum('wtnj,wtnk,wtn->jk', basis.conj(), basis,
                           cache['residual_weights'].to(basis.dtype)).real
        ridge = float(config.get('preconditioner_floor', 1e-10))*gram.diag().mean()
        chol = torch.linalg.cholesky(gram+ridge*torch.eye(self.krylov_terms, dtype=gram.dtype, device=gram.device))
        self.preconditioner.copy_(torch.linalg.solve_triangular(chol.T, torch.eye(
            self.krylov_terms, dtype=gram.dtype, device=gram.device), upper=True))
        self._training_cache = cache

    def _make_basis(self, width):
        g, h = self._geometry(width), self._duhamel_matrix(width)
        seeds = self._seeds(g, h)
        return self._basis_from_seeds(seeds, g, h)

    def _basis_from_seeds(self, seeds, g, h):
        fields = []
        for j in range(self.krylov_terms):
            value = (seeds[j//self.chain_size]/self.seed_norms[j//self.chain_size]
                     if j%self.chain_size == 0 else self._operator(fields[-1], g, h))
            value = value-sum(self.recurrence[i, j]*fields[i] for i in range(j))
            fields.append(value/self.recurrence[j, j])
        nodal = torch.stack(fields, dim=-1)
        return torch.einsum('jt,tnk->jnk', self.time_transform.to(nodal.dtype), nodal)

    def basis(self, width):
        if width.requires_grad:
            return WidthBasis.apply(width, self)
        key = (float(width), str(width.device), torch.complex128 if width.dtype == torch.float64 else torch.complex64)
        if key not in self._basis_cache:
            with torch.no_grad():
                result = self._make_basis(width)
            if len(self._basis_cache) >= self.cache_widths:
                self._basis_cache.pop(next(iter(self._basis_cache)))
            self._basis_cache[key] = result
        return self._basis_cache[key]

    def feedback_amplitudes(self, t, width):
        shared = self.preconditioner@self.total_coefficients()
        if width.requires_grad:
            return torch.stack([self._feedback_one(tt, ww, shared) for tt, ww in zip(t, width)])
        unique, inverse = torch.unique(width, return_inverse=True)
        result = torch.zeros(len(t), self.feedback_modes+1, device=t.device, dtype=torch.complex128)
        for i, w in enumerate(unique):
            mask = inverse == i
            poly = self.basis(w)@shared.to(torch.complex128)
            blocks = []
            for tt in t[mask].split(self.query_block):
                values = chebyshev(2*tt/T_MAX-1, self.time_degree).to(poly.dtype)
                blocks.append((values@poly)*(tt/T_MAX)[:, None].square())
            result[mask] = torch.cat(blocks)
        return result

    def _feedback_one(self, t, width, shared):
        poly = self.basis(width)@shared.to(torch.complex128)
        value = chebyshev(2*t/T_MAX-1, self.time_degree).to(poly.dtype)
        return (value@poly)*(t/T_MAX).square()

    def forward(self, points):
        if len(points) == 0:
            return points.new_empty((0, 2))
        x, t, width = points[:, 0], points[:, 1], points[:, 2]
        if not points.requires_grad:
            tw, inverse = torch.unique(points[:, 1:3], dim=0, return_inverse=True)
            rad = self.radiation_amplitudes(tw[:, 0], tw[:, 1])[inverse]
            feed = self.feedback_amplitudes(tw[:, 0], tw[:, 1])[inverse]
        else:
            rad = self.radiation_amplitudes(t, width)
            feed = self.feedback_amplitudes(t, width)
        cr = torch.cos(math.pi*x[:, None]*self.radiation_n/width[:, None])
        cf = torch.cos(math.pi*x[:, None]*self.feedback_n/width[:, None])
        mr = torch.where(self.radiation_n == 0, 1.0, 2.0)
        mf = torch.where(self.feedback_n == 0, 1.0, 2.0)
        q = (line_field(x[:, None], t[:, None], width[:, None])
             +(cr*rad*mr).sum(dim=-1)+(cf*feed*mf).sum(dim=-1))
        return torch.stack((q.real, q.imag), dim=-1)
