# HJ observations

`observations.npz` contains noisy observations.

There are 5,945 training samples at frames 1–29 (times 1/98–29/98),
and 1,025 validation samples at frames 30–34 (30/98–34/98).
Each frame observes 205 spatial points. Points are `(x,y,t)`; values are scalar.

- Gaussian noise at 3% of training clean RMS.

Only these four arrays are exported. There are no clean fields,
coefficient labels, future fields, fitted states or previous candidates here.
The validation observations are public selection data, not a blind test.
