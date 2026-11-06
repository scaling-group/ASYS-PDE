"""Frozen-field metrics: periodic spatial means and time trapezoids."""
import numpy as np
import torch


def pool(rows):
    error, field, correction = (
        sum(row[key] for row in rows)
        for key in ('squared_error', 'reference_energy', 'reference_correction_energy')
    )
    if min(field, correction) <= 0 or not np.isfinite([error, field, correction]).all():
        raise ValueError('Invalid pooled energies')
    return dict(field_relative=float(np.sqrt(error / field)),
                correction_relative=float(np.sqrt(error / correction)),
                squared_error=error, reference_energy=field,
                reference_correction_energy=correction)

def breather(x, times):
    # Declared infinite-line analytical comparator, never a search input here.
    tt, xx = np.meshgrid(times, x, indexing='ij')
    numerator = np.cosh(3 * xx) + 3 * np.exp(4j * tt) * np.cosh(xx)
    denominator = np.cosh(4 * xx) + 4 * np.cosh(2 * xx) + 3 * np.cos(4 * tt)
    return 4 * np.exp(.5j * tt) * numerator / denominator


def predict(model, x, times, width):
    tt, xx = np.meshgrid(times, x, indexing='ij')
    points = torch.tensor(np.column_stack((xx.ravel(), tt.ravel(), np.full(xx.size, width))), dtype=torch.float64)
    with torch.no_grad():
        uv = torch.cat([model(p) for p in points.split(4096)])
    if uv.shape != (xx.size, 2) or not torch.isfinite(uv).all():
        raise ValueError('invalid frozen field')
    return (uv[:, 0] + 1j * uv[:, 1]).numpy().reshape(tt.shape)


def metrics(predicted, reference, baseline, times, width):
    def energy(value):
        return float(np.trapezoid(np.mean(abs(value) ** 2, axis=1), times))
    error = energy(predicted - reference)
    field_energy, correction_energy = energy(reference), energy(reference - baseline)
    if min(field_energy, correction_energy) <= 0 or not np.isfinite(error):
        raise ValueError('invalid norm denominator or field error')
    mass = 2 * width * np.mean(abs(predicted) ** 2, axis=1)
    target_mass = 8 * np.tanh(width)
    if mass[0] <= 0 or not np.isfinite(mass).all():
        raise ValueError('invalid sampled mass')
    return dict(field_relative=np.sqrt(error / field_energy),
                correction_relative=np.sqrt(error / correction_energy),
                squared_error=error, reference_energy=field_energy,
                reference_correction_energy=correction_energy,
                mass_initial_relative=abs(mass[0] - target_mass) / target_mass,
                mass_drift_relative=float(np.max(abs(mass - mass[0])) / mass[0]),
                mass_target_relative=float(np.max(abs(mass - target_mass)) / target_mass))

