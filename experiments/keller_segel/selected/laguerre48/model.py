"""Two radial shape modes, warmly initialized from solver_e4da6f5b83fa.

u=9.5 sum_j w_j/a_j exp(-q_j)(1+b4_j L4(q_j)+b8_j L8(q_j))^2/Z_j,
q_j=r^2/a_j, Z_j=1+b4_j^2+b8_j^2. Both shapes start cubically in time.
Laguerre orthogonality gives unit kernel mass and mean q
(1+9*b4^2+17*b8^2)/Z. L4 and L8 have no mass or first-moment cross term.
The actual kernel means enter the inherited width normalization: radial mass
4.75 and second radial moment 4.75-3.5625*t are structural whole-plane
identities. Finite-radius decay/flux corrections still require validation.
The inherited Gaussian initial value, PDE tangent and second time jet persist.
No singularity clock, trajectory table or time-stepping is embedded.

Changes: warm start from e4's exported 325 fitted parameters instead of its
older cfc initialization; add 13 independent cubic L8 shape coefficients.
Retain 13 kernels, degree-12 temporal coefficients, L4 bound .02, raw area
loss, radial/time quadrature and damped Gauss--Newton. L8 bound .005.
This is a radial-family extension plus a disclosed warm continuation, not a
controlled comparison of guidance alone or an already successful new fit.
338 trainable scalars; 375 persistent buffer scalars. Inherited fitted
initialization is 637 scalars: 312 frozen temporal seed, 312 trainable temporal
corrections and 13 trainable L4 shapes. L8 coefficients start at zero.
Constructor only allocates arrays and exact fixed polynomial tails; it never
reads an artifact, evaluates an objective, or fits. All fitted field quantities
are in state_dict. Numerical optimization remains evaluator-owned.

Evidence retained in the immutable examples: e4 public physics 1.66665712e-8,
796 steps and 1679 closures; running quarter best training losses
4.76813824e-8,1.80647254e-8,1.66632512e-8,1.61978802e-8,1.60373042e-8.
It was still improving at its allowance boundary, so class deficiency is not
established. The force-conditioned Gaussian example 2d869f3858ea plateaued
at 1.84996450e-8 training and scored physics 4.47871568e-8; a loss metric
change is not automatically a raw-score improvement. Other inherited adverse
fits remain available: first-jet LBFGS physics 1.70488464e-4, hazard LBFGS
2.13504882e-4 and alternate family 2.73457510e-4.

No-step training-quadrature tangent diagnostics at the fitted e4 state:
max normalized residual correlation .00129032 over current columns;
independent cubic-onset L2/L6/L8 columns .000387815/.00424331/.0194763.
Shared L8 correlation -.000755348. These linear diagnostics motivate the
independent L8 extension; they do not prove nonlinear improvement. At 121
independent uniform times, e4 raw area MSE mean 1.53129398e-8, maximum
4.56823093e-8 at t=.158862, endpoint 2.95453100e-8, u(0,T)=625.191970.
These are inherited synthetic readouts, not a numerical reference solution.

Exponentially weighted Horner evaluation avoids 0*infinity for remote q.
Forward uses ordinary coordinate/parameter AD; the separate exact analytic
training VJP supplies only the first loss derivative. Higher parameter AD
remains available through forward and differential_parts. Polynomial tails
are constructed with rational arithmetic, then stored as persistent buffers.

Live primary sources read in this phase:
Blanchet, Dolbeault, Perthame (2006), mass/second-moment identities, converted
from whole-plane mass 2*pi*m; it supplies no trajectory for this Gaussian IVP:
https://ejde.math.txstate.edu/Volumes/2006/44/blanchet.pdf
NIST DLMF 18.3 and 18.9, Laguerre orthogonality and recurrence, adapted here
into a positive squared-polynomial kernel, not a supplied Keller--Segel ansatz:
https://dlmf.nist.gov/18.3 ; https://dlmf.nist.gov/18.9
Rathore et al. (ICML 2024), differential residual conditioning; the retained
optimizer is not their NNCG algorithm:
https://proceedings.mlr.press/v235/rathore24a.html
The inherited Springer source fetch timed out; no numerical target was used.

Final synthetic validation, ASYS_NUM_THREADS=2, no optimizer steps or manual fit:
warm values/first coordinate gradients agree with fitted e4 on 96 radii x 47
times to 2.28e-13/1.23e-10 absolute. Ordinary AD u_t/diffusion/chemistry agree
with the analytic jets to 1.17e-10 absolute for warm and deterministic nonzero
perturbations. Initial u/u_t/u_tt errors are <=1.8e-15/7.2e-14/4.5e-11.
On a separate 160-radius x 11-time grid, the analytic training VJP versus direct
radial backpropagation differs by <=2.9e-12 absolute (relative L2 3.79e-10 at
warm, 2.24e-15 perturbed). Factored full-grid gradients agree to <=1.8e-13.
The residual Jacobian directional product versus ordinary parameter forward
AD has scaled error <=1.05e-13; boundary Jacobians <=4.8e-15; Gram contraction
relative error <=4.4e-16. Ordinary second parameter derivatives remain finite.
Strict state_dict transfer to a fresh interpreter reproduces all protected
value, no-grad value, dr, dt, drr and initial-value probes exactly. The cold
child imports neither loss.py nor optim.py.

Independent 12-node quadrature in 174 radial cells at 101 uniform times checks
warm, small-perturbation and near-bound shapes: finite nonnegative fields,
radial mass/second-moment errors <=2.7e-15 and exact origin symmetry. Independent
128-node partial-kernel integrals agree with the stored tails within 1.8e-13.
All-positive 1e6 temporal/shape coefficients also give finite field/coordinate
jets in the sampled window; this is a stress sample, not a global error bound.
Warm boundary loss at 101 uniform times is 1.81049e-176.

Adverse physical diagnostic retained: monotonicity is NOT structural. A 1200
log-radius x 101-time ordinary-AD sample finds positive u_r in inherited e4's
outer tail, maximum 1.61871e-6 at r=7.13688,t=T,u=3.66921e-7; the largest density
on positive-slope samples is 8.74187e-7. The warm extension reproduces this.
A near-bound shape stress case raises these to u_r=6.11593e-6 and maximum
positive-slope density 3.27427e-6. These rings are a compatibility limitation,
even with small area residuals and exact moments; no solution is certified.

Retained development failures: initial generated method indentation prevented
import and was corrected before derivative validation. An initial synthetic
second-jet comparison reported error 63.57 because its reference incorrectly
multiplied the entire chemotactic initial flux by r^2. Correct reference flux
is M_t=-2*r^2*u0+M0*u0; the corrected true-jet check passes as reported above.
A central finite-difference Jacobian test had scaled error 4.1e-6 from residual
subtraction; the independent ordinary-AD directional test resolves that check.

Measured import/construction .0181 s, optimizer setup 1.6533 s, with no objective
evaluation during setup. Configured synthetic loss/backward .343-.379 s and
normal-matrix assembly .463-.566 s. Initial objective 1.60373041889179e-8 agrees
with inherited 1.60373041888739e-8; new regularization is zero at initialization.
These timings establish feasibility, not whole-fit convergence or query speed.
The added algebra may reduce the number of native steps; improvement under the
fixed allowance is untested. The inherited fit still improving is an equally
plausible reason for further gains. Low residuals and added modes are no proof.
"""
from __future__ import annotations

import math
from fractions import Fraction

import torch
from torch import nn

DTYPE = torch.float64
T_MAX = 0.86652

# Best public fitted state SHA256: 4aa9970f8cf77f7097ccbd13d706ad20d10a704949135e02ef3666b73b807754
# Frozen seed is inherited unchanged; runtime never reads this artifact.
_INHERITED_TIME_COEFFICIENTS = [
    [5.649104765968343, -3.13057529991976, -3.0300526733111135, 3.278459811348018, 0.4744677976257525, -4.007649649700406, 5.06858689138379, -4.1039605409468205, 2.5453525394940537, -1.180074486930304, 0.36345547481063034, -0.036070502361823964, -0.012290405898728101],
    [0.46117743696742486, -1.5268226656488395, 0.07616497579975812, 1.0774656732014645, -0.5444984833725037, -0.1856570158083153, 0.08528656667298269, 0.28209305754220715, -0.29344224916501915, 0.22775387050125878, -0.17686380389475528, 0.1149064259299176, -0.03570174813482549],
    [1.4083617691548598, 0.06448088087182804, -2.716776107028523, 0.20280520438202054, 2.189342328351094, -1.857478727152095, 0.5498133563386661, 0.3877577805419008, -0.6456868238235689, 0.29484357514266685, 0.024920518110367135, -0.07738994973387409, 0.028627399770425544],
    [-3.7651991011941432, 5.613981349836546, -3.7054517206102373, -0.30170924187354325, 2.96012214386603, -3.232032788546677, 2.4284456966108996, -1.6173993100261794, 1.0524533353827012, -0.6867375250549221, 0.4174847480741519, -0.1920064770404891, 0.04454718401402408],
    [7.347672289497289, -0.7571722342594174, -7.996537727033364, 5.346363222156002, 1.0789171201196173, -3.5444297107632696, 2.2504236065939796, -0.0041519989252905894, -1.2887343847271062, 1.1244907783321423, -0.5099285993376035, 0.1402587979049684, -0.01741730201637143],
    [-1.2889973587865953, 0.07389546576355543, 1.163239742099607, -1.1407888215123836, 0.012177587584457323, 1.146597052154772, -1.304469004551045, 0.5564825705873114, 0.1838348568428941, -0.477904246347153, 0.3594451773691668, -0.1329076452088803, 0.02167379255773299],
    [-1.5216177396005086, -0.5248082020986187, 0.9491729147539469, 0.14477926684115988, -0.6801372834731523, 0.5513947439532019, -0.04173253283759122, -0.544244292539457, 0.8350715297930688, -0.811306672502468, 0.4608454906939485, -0.12329129615975087, 0.00904050379501819],
    [-2.2294811612058827, -0.2521989612060005, 2.2124734439894302, -0.05715156312900596, -2.1477710396492915, 1.797914574386341, 0.5479732608691124, -2.3121309033194963, 2.2875720221422737, -1.462007582281437, 0.6391769346960633, -0.16135389561793978, 0.015768793225156504],
    [3.5226923838816324, -2.9724785449574944, 0.2558295660881018, 0.31603445192569574, -0.011133032408018764, -0.43031838637777603, 1.0290447750391307, -1.3161830227470808, 1.163336328082697, -0.8795710844220266, 0.5498555232440194, -0.24410425402985048, 0.05733221330345519],
    [2.6007725668725357, -4.4015210000733145, 3.1478346770688868, -3.0133890537813692, 2.738810479826333, -2.268310938355683, 1.8003824139207139, -1.380484120615954, 0.9486126264098486, -0.5658029140801873, 0.2627020292071597, -0.07618397899437583, 0.010740397764499213],
    [0.05794536354134253, 0.011188201160468373, -0.15404939954543734, -0.6582352153391389, 1.0478994506492445, -0.9666629379901827, 0.8126565509496133, -0.7656349344001859, 0.6234826173855447, -0.42762825333122834, 0.2042021393636151, -0.04901174554254665, 0.0040999016075715196],
    [0.4119681867457054, -0.12556864659915123, 0.9958031402405862, -2.1508234515881477, 2.2150898620657027, -1.6485481964099669, 1.1139721824819633, -0.822160263415391, 0.5778113095257555, -0.38634731657574, 0.17866320068083008, -0.03646912923722569, 0.0010974384193251226],
    [0.5933362455422648, -0.4385929076261383, 1.941817765620225, -3.5742372789529004, 3.404626453388651, -2.2617695409917467, 1.357007126010036, -0.8083606153211489, 0.40831129629585033, -0.24705905280966578, 0.10143325519540108, 0.002587707626313782, -0.010529346628135795],
    [-8.63811502276753, 17.22578674643451, -13.04243831268303, 8.202077675425821, -5.305245730214655, 3.934617013222778, -2.7801829305885306, 1.688521111016418, -0.9824203079202484, 0.4504201275083042, -0.17072030694247384, 0.07169167856548679, -0.020227546521384095],
    [-1.158170779742966, 5.969287252137191, -7.596365610080051, 7.380020639518619, -6.698749811162199, 6.286617642917337, -5.847794937493062, 4.901979305071366, -3.491775889344559, 1.9782950417172087, -0.8654523948400645, 0.28998125982922407, -0.056407760578443246],
    [4.380389498188712, -5.681786651336288, 1.8233793383953156, -0.0180006580220519, -0.18619387246695362, -0.4045323278079576, 0.9345697437900616, -0.8527631408640933, 0.34481109183480163, -0.09268873101473882, 0.020586692234650705, 0.008309181003273505, -0.0015043372205861727],
    [2.6887892722211464, -4.246558001405749, 0.3808570375522825, 1.4044893960209806, -1.2592900649236094, 0.641795506937762, -0.4254797728599975, 0.578182738564815, -0.7082186995427905, 0.5608838943542322, -0.2897505821152265, 0.107654024029157, -0.02305849859284393],
    [-2.0830440649263258, 1.6856072466453624, 2.1913876527462626, -1.6303087567601398, -0.01574441915143153, 0.8569994513347979, -0.8226287246427401, 0.06881595995650248, 0.846414082574923, -1.008817417490611, 0.5991339091731334, -0.1831883332004851, 0.02071522047890579],
    [0.9437023430947952, 2.5669544853030573, -0.22891228319380943, -0.9990699924878373, 2.1206631170505426, -1.9905422818513294, 0.7177631565416271, 1.2414717462507452, -2.2653092365210936, 2.0015761615001626, -1.099675825162897, 0.3963176474418479, -0.08353908975328167],
    [2.6397822188940543, -1.1795329201905762, 0.3288296564795036, -1.387997916317124, 0.9445618827047337, -0.11247496346581007, -0.07160966889683289, 0.15681792102361866, -0.34244602383833134, 0.18703476062709068, -0.06193140774000067, 0.03865531304019829, -0.010464358647996018],
    [1.7778791465208987, -0.14963316361664486, -0.591852817691742, 0.16423741674915227, -0.5700524596688038, 0.9299083357632757, -0.8566496549173648, 0.7760357301126996, -0.6728023754136793, 0.36739367141394097, -0.17911757724895402, 0.08416886648974525, -0.02252030918512387],
    [0.5655699880929626, 0.5512501072449144, -0.9061993296848457, 0.5678430814372055, -0.6141401347542191, 0.6958852040500869, -0.5152022451862265, 0.3663711005425839, -0.29407794676705595, 0.15513622920261033, -0.062563572124264, 0.025375657540405744, -0.0064235385917075375],
    [0.35035876842464714, -1.6010369284141883, 1.0877834901908605, -0.5752669187087419, 0.2325614031837326, 0.1757118279348294, -0.29348061213121296, 0.45340900951245783, -0.4578910824707723, 0.30059820011313376, -0.19652833654871552, 0.13770921580173257, -0.04979082730476095],
    [3.214032127084134, -4.985629226716379, 2.3334075736220585, -1.6281833482781407, 1.5136453026386711, -0.4797899836881348, 0.10752333043045463, 0.0738918435323011, -0.2226818582434239, 0.21525900176655494, -0.08126055431553318, 0.06974135789021778, -0.05621585024904885],
]
_INHERITED_CORRECTIONS = [[-0.14044747826973208, 1.5325698348358534, -3.6373935115203135, 4.5517107327283295, -3.065815895817645, 0.7359744970598784, 0.9572254720121126, -1.4480982428931592, 1.1127479215465768, -0.6268865345618682, 0.27275090547698416, -0.09422133210394366, 0.020316466165887], [0.22486697050392007, -0.2831219327721094, 0.19931629348247556, 0.0729691788480608, -0.04936146273395728, -0.01898635749902662, 0.1457051229006977, -0.18047072143584, 0.1112033482648795, -0.08579039362781612, 0.06725821186398948, -0.04393894390471365, 0.01476287432284469], [0.7678365319601692, -1.981455070593196, 2.714965249087459, -3.706772754898847, 4.044717514235546, -3.456980536219931, 2.1080769791024325, -0.8624871825283961, 0.18570469478807936, 0.10166690134250608, -0.12298227235467252, 0.053627262243957564, -0.01323890854864728], [-1.0080854103134487, 2.277041480554005, -2.7761967847756095, 3.0259836880929303, -2.8747413664721706, 2.376492777566771, -1.83877935405293, 1.3800211525867783, -0.9762345268771829, 0.620489964259309, -0.32694890731287085, 0.12654895139134142, -0.02623638822386967], [-1.0643070684595433, 2.5178930982047354, -3.615539146206247, 3.8221390237809274, -3.1279539463459343, 2.0058196697020167, -1.1640440204023437, 0.5375942798982706, -0.06290441902878346, -0.07666961791349067, 0.061165842456643346, -0.028607773870796143, 0.004963653506106727], [1.0689318840826254, -1.8560582010873274, 0.5922821047621762, 1.8649678327714123, -4.451292804619848, 5.833741494447532, -5.4669311742616635, 3.9124455791014685, -2.1838605926318206, 0.9553007555604686, -0.32257745444677327, 0.07985516189096484, -0.012366021154821834], [0.06190014354930856, -0.9479018628781886, 1.873743292950477, -1.5166741584713284, 1.0363024571042652, -1.347454772670873, 1.8209867191031859, -1.7213669214157221, 1.0455643078205281, -0.36033029425826724, 0.05314895699679694, -0.005745702090008355, 0.0033628369033347383], [-0.3718035496147576, 0.35663975438424733, 0.22177645432793486, -0.2257827763476844, -0.5903653025783361, 1.0908751584426715, -0.36616727967116103, -1.1204142784615698, 2.064477194667817, -1.8323404598439237, 0.9973418430196788, -0.3335496632001009, 0.05548468028909251], [1.4811939685573225, -2.7548080027074953, 2.153640509860259, -1.4507027768227632, 0.6425723116629641, -0.0010329876089355235, -0.47358684308951154, 0.6677952740768229, -0.6172890840033208, 0.47823053377261837, -0.285653160813385, 0.11696027287891508, -0.026824620466126527], [-0.9237951021833294, 1.812682178825979, -1.7214484777906525, 1.5734205527251397, -1.351408335255647, 1.1697212136424022, -1.022616503945866, 0.8341891704604423, -0.5870552162574052, 0.34273773400405944, -0.15576088741785776, 0.04897679379218311, -0.00867898985002979], [-1.5486882789796097, 2.985959123147487, -2.7614248986866934, 2.406944334681377, -1.8785456133465082, 1.4152208175862315, -1.0857499106554078, 0.7899735498265353, -0.49025841424974254, 0.24506099532530812, -0.09728011597219341, 0.02814991625941268, -0.004837610936851351], [-2.1834669469888883, 4.2109935346597345, -3.8601897765729287, 3.2382931476637236, -2.373000109514879, 1.6627488116565423, -1.1857588689256653, 0.7799500875548356, -0.4399007317921218, 0.2055030987820779, -0.07225722599536008, 0.01424323998896809, -0.0009943087733934365], [-0.026227200017569754, 0.08215944723854272, -0.11129396483436987, -0.05342269741129086, 0.3097015069714387, -0.3924798942013465, 0.27014038053587447, -0.17809694606464468, 0.13183746175735186, -0.08426367083638696, 0.06400281630382687, -0.04451513184944439, 0.013564531833502464], [-1.474050271397578, 3.03415357798374, -2.9495937806824144, 2.5654466207861497, -2.0779353318917546, 1.5915081258134547, -1.2842502716607695, 1.0226633736911432, -0.6649940227057844, 0.3701153547088743, -0.19090786156078124, 0.06507218614287943, -0.005894859400816471], [0.17359393836690873, -0.651098011227859, 1.1618495523105794, -1.305892503572169, 1.0642978303917752, -0.5275778439979661, 0.1928173451050531, -0.17247185559492698, 0.2642788023368599, -0.2448839313038315, 0.13151968458902658, -0.049315572915548625, 0.012383758792378149], [-1.2282697990496096, 2.0183899232667404, -1.2741887311264324, 0.8485171296942702, -1.0728392768425477, 1.526625939181857, -1.641687264391973, 1.210741100694417, -0.5828794419785214, 0.21637861364238442, -0.07452957765370595, 0.0195581529155695, -0.004006238386898152], [-0.9147837058528698, 1.4232800305751603, -0.5558079485613336, 0.19764064541422605, -0.49614086260227147, 1.138378644110823, -1.402942815349415, 1.152163230998802, -0.686944215605539, 0.3299543536359627, -0.15360953108480158, 0.05589726516676994, -0.008095351145302031], [-1.5085692833218822, 1.9149253287125245, 0.6722491086048881, -1.8081348728144617, 2.1071919994870636, -2.027625868514394, 2.3825956752303914, -2.409823470100246, 1.4441570555257826, -0.42264107429490755, -0.04878770996584343, 0.07466877100480558, -0.01320435863236766], [-0.9481892028692073, 2.976019429597791, -2.152039829475504, 0.7556252547898967, 1.5094662817756292, -3.1256887570250536, 5.196845405353695, -6.934938887716507, 6.635375326063629, -4.625134259068821, 2.160265148558977, -0.6379121880065806, 0.11017107237599712], [0.03958917059309221, -0.13520314327637728, -0.20027804182939435, 0.13171052119615423, 0.04282838169826219, -0.26886814854952173, 0.34736327718745263, -0.42270943738078537, 0.4420201678950458, -0.24139388306722667, 0.11694573916099124, -0.07126328024904854, 0.018753404726730077], [0.6050534111302832, -1.2414876988082986, 0.9257917045720857, -0.9726333319229882, 0.8416293559321223, -0.7231927984635356, 0.5316729565950431, -0.4305889667820533, 0.34623622870155546, -0.16046923704554444, 0.08964810679356236, -0.07083504090975616, 0.023903781512938874], [0.9691007242917393, -1.9571968644189885, 1.7797487581242464, -1.5720014750009568, 1.2557477277595597, -1.0450322485159194, 0.7823411763902222, -0.536206706911662, 0.34444668912059434, -0.17416338698082812, 0.1063410542996349, -0.06090918424703038, 0.01412683466608497], [0.12421687152389402, -0.294585271174146, 0.7784320478544868, -0.25253093380148156, 0.5484908521922673, -0.6732626696011274, 0.3929548872291678, -0.3398704239716877, 0.31287431563631785, -0.20409639573336646, 0.08529587162557939, -0.05070668790412339, 0.023116597312971075], [0.27437990510469545, 0.29857631320330913, 0.4280551837631478, -0.1665101810769683, 0.31917023149636026, -0.10339046106389076, 0.3043806219064732, -0.13058560889664192, 0.05247894578633224, -0.27314264956202733, 0.066069810300477, -0.04836942529311943, 0.080515288769254]]

_INHERITED_SHAPES = [-0.0013941811575833686, -0.0026480373824500384, -0.0020639174158972287, -0.013326255936524384, -0.0008566124863417005, -0.00027358296132247076, 0.000789587578492023, -0.003265581627136515, -0.0009503161995115531, -0.0005785254105199395, -0.0002831983898765453, -8.480465044607033e-05, 0.0010899964667993216]


class Candidate(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        self.n_core = int(config.get("n_core", 12))
        self.n_first = int(config.get("n_first", 6))
        self.degree = int(config.get("time_degree", 12))
        if (self.n_core, self.n_first, self.degree) != (12, 6, 12):
            raise ValueError("The inherited field uses two groups of six and degree twelve")
        self.register_buffer("inherited_time_coefficients", torch.tensor(
            _INHERITED_TIME_COEFFICIENTS, dtype=DTYPE))
        self.register_buffer("correction_bound", torch.tensor(6.0, dtype=DTYPE))
        for name, default in (("shape_bound", .02), ("shape8_bound", .005)):
            bound = float(config.get(name, default))
            if not 0 < bound <= .05:
                raise ValueError("Shape bounds must lie in (0, .05]")
            self.register_buffer(name, torch.tensor(bound, dtype=DTYPE))
        self.time_coefficients = nn.Parameter(torch.tensor(_INHERITED_CORRECTIONS, dtype=DTYPE))
        self.shape_coefficients = nn.Parameter(torch.tensor(_INHERITED_SHAPES + [0.0]*13, dtype=DTYPE))
        self.register_buffer("log_rates", torch.cat([
            torch.linspace(math.log(.04), math.log(10.), count, dtype=DTYPE)
            for count in (self.n_first, self.n_core-self.n_first)]))
        def laguerre(n):
            return [Fraction((-1)**k*math.comb(n,k), math.factorial(k)) for k in range(n+1)]
        def product(a, b):
            return [sum((a[i]*b[k-i] for i in range(len(a)) if 0<=k-i<len(b)), Fraction(0))
                    for k in range(len(a)+len(b)-1)]
        def register_tail(name, polynomial, norm, moment):
            assert sum(v*math.factorial(k) for k,v in enumerate(polynomial)) == norm
            assert sum(v*math.factorial(k+1) for k,v in enumerate(polynomial)) == moment
            tail = list(polynomial)
            for k in range(len(tail)-2, -1, -1):
                tail[k] += (k+1)*tail[k+1]
            assert tail[0] == norm
            self.register_buffer(name, torch.tensor([float(v) for v in tail[1:]], dtype=DTYPE))
        l4, l8 = laguerre(4), laguerre(8)
        register_tail("tail_laguerre", l4, 0, 0)
        register_tail("tail_square", product(l4,l4), 1, 9)
        register_tail("tail_laguerre8", l8, 0, 0)
        register_tail("tail_cross", product(l4,l8), 0, 0)
        register_tail("tail_square8", product(l8,l8), 1, 17)

    def time_basis(self, tau, derivatives=False):
        return self._chebyshev(tau, self.degree, derivatives)

    @staticmethod
    def _chebyshev(tau, degree, derivatives=False):
        z = 2 * tau - 1
        basis = [torch.ones_like(z)]
        slope = [torch.zeros_like(z)]
        if degree:
            basis.append(z)
            slope.append(torch.full_like(z, 2.0))
        for _ in range(2, degree + 1):
            basis.append(2 * z * basis[-1] - basis[-2])
            if derivatives:
                slope.append(4 * basis[-2] + 2 * z * slope[-1] - slope[-2])
        if not derivatives:
            return torch.cat(basis, dim=1)
        return torch.cat(basis, dim=1), torch.cat(slope, dim=1)

    def raw_curves(self, times, derivatives=False):
        tau = times.reshape(-1, 1).to(dtype=DTYPE) / T_MAX
        if not derivatives:
            return self.time_basis(tau) @ self.time_coefficients.T
        basis, slope = self.time_basis(tau, True)
        return basis @ self.time_coefficients.T, slope @ self.time_coefficients.T

    def temporal_components(self, times, derivatives=False):
        if derivatives:
            raw, raw_tau = self.raw_curves(times, True)
            return self.components_from_raw(times, raw, raw_tau, self.shape_coefficients)
        return self.components_from_raw(times, self.raw_curves(times), shape=self.shape_coefficients)

    def components_from_raw(self, times, raw, raw_tau=None, shape=None):
        """Same temporal algebra with explicit polynomial values/slopes.

        This separation permits a small exact training Jacobian. Forward
        values and coordinate derivatives always use ordinary PyTorch AD.
        """
        t = times.reshape(-1, 1).to(dtype=DTYPE)
        tau = t / T_MAX
        if shape is None:
            shape = self.shape_coefficients
        seed_basis, seed_slope = self._chebyshev(tau, 12, True)
        seed = seed_basis @ self.inherited_time_coefficients.T
        seed_squash = torch.tanh(seed / 12.0)
        delta_squash = torch.tanh(raw / self.correction_bound)
        correction = 12 * seed_squash + self.correction_bound * delta_squash
        shape = shape.reshape(1, 2, self.n_core + 1)
        bounds = torch.stack((self.shape_bound, self.shape8_bound))[None, :, None]
        shape_level = (bounds * torch.tanh(shape / bounds)).flatten(1)
        beta = tau.pow(3) * shape_level
        beta_t = 3 * tau.square() * shape_level / T_MAX
        b4, b8 = beta.split(self.n_core + 1, dim=1)
        norm = 1 + b4.square() + b8.square()
        mean_q = (1 + 9*b4.square() + 17*b8.square()) / norm
        if raw_tau is not None:
            correction_tau = ((1 - seed_squash.square()) *
                              (seed_slope @ self.inherited_time_coefficients.T)
                              + (1 - delta_squash.square()) * raw_tau)
        f1, f2 = correction[:, :1], correction[:, 1:2]
        scale = correction[:, 2:self.n_core + 2]
        logits = correction[:, self.n_core + 2:]
        # Each softmax has one fixed pivot, removing both logit gauges.
        conditional = []
        conditional_t = []
        offset = 0
        start = 0
        for count in (self.n_first, self.n_core - self.n_first):
            pivot = count // 2
            local = logits[:, offset:offset + count - 1]
            local = torch.cat((local[:, :pivot], torch.zeros_like(t), local[:, pivot:]), dim=1)
            rates = self.log_rates[start:start + count]
            prob = torch.softmax(local + rates - rates.exp(), dim=1)
            conditional.append(prob)
            if raw_tau is not None:
                local_t = correction_tau[:, self.n_core + 2 + offset:
                                         self.n_core + 2 + offset + count - 1] / T_MAX
                local_t = torch.cat((local_t[:, :pivot], torch.zeros_like(t),
                                     local_t[:, pivot:]), dim=1)
                conditional_t.append(prob * (local_t - (prob * local_t).sum(1, keepdim=True)))
            offset += count - 1
            start += count

        amplitude = 9.5
        denominator = 1 + (amplitude + 2) * t
        exp1, exp2 = torch.exp(tau.square() * f1), torch.exp(tau * f2)
        transfer1 = amplitude * t / denominator * exp1
        transfer2 = 1.5 * amplitude ** 2 * t.square() / denominator * exp2
        normalizer = 1 + transfer1 + transfer2
        fraction1, fraction2 = transfer1 / normalizer, transfer2 / normalizer
        weights = torch.cat((normalizer.reciprocal(), fraction1 * conditional[0],
                             fraction2 * conditional[1]), dim=1)
        raw_halo = 1 + 4 * t
        shifted = scale - self.log_rates
        raw_core1 = 0.5 * (1 + 6 * t) * torch.exp(tau.square() * shifted[:, :self.n_first])
        raw_core2 = (1 / 3) * torch.exp(tau * shifted[:, self.n_first:])
        raw_core = torch.cat((raw_core1, raw_core2), dim=1)
        raw_widths = torch.cat((raw_halo, raw_core), dim=1)
        mean_raw_width = (weights * raw_widths * mean_q).sum(1, keepdim=True)
        second_moment = 1 - 0.75 * t
        widths = raw_widths * second_moment / mean_raw_width
        if raw_tau is None:
            return weights, widths, beta

        f1_tau, f2_tau = correction_tau[:, :1], correction_tau[:, 1:2]
        scale_tau = correction_tau[:, 2:self.n_core + 2]
        transfer1_t = amplitude * exp1 / denominator.square()
        transfer1_t = transfer1_t + transfer1 * (2 * tau * f1 + tau.square() * f1_tau) / T_MAX
        transfer2_t = 1.5 * amplitude ** 2 * t * (2 + (amplitude + 2) * t) * exp2 / denominator.square()
        transfer2_t = transfer2_t + transfer2 * (f2 + tau * f2_tau) / T_MAX
        normalizer_t = transfer1_t + transfer2_t
        fraction1_t = transfer1_t / normalizer - fraction1 * normalizer_t / normalizer
        fraction2_t = transfer2_t / normalizer - fraction2 * normalizer_t / normalizer
        weights_t = torch.cat((-normalizer_t / normalizer.square(),
                               fraction1_t * conditional[0] + fraction1 * conditional_t[0],
                               fraction2_t * conditional[1] + fraction2 * conditional_t[1]), dim=1)
        halo_log_t = 4 / (1 + 4 * t)
        core1_log_t = 6 / (1 + 6 * t) + (2 * tau * shifted[:, :self.n_first]
                       + tau.square() * scale_tau[:, :self.n_first]) / T_MAX
        core2_log_t = (shifted[:, self.n_first:] + tau * scale_tau[:, self.n_first:]) / T_MAX
        core_log_t = torch.cat((core1_log_t, core2_log_t), dim=1)
        raw_log_t = torch.cat((halo_log_t, core_log_t), dim=1)
        b4t, b8t = beta_t.split(self.n_core + 1, dim=1)
        norm_t = 2*(b4*b4t + b8*b8t)
        mean_q_t = (18*b4*b4t + 34*b8*b8t - mean_q*norm_t) / norm
        mean_raw_t = (raw_widths * ((weights_t + weights * raw_log_t) * mean_q
                     + weights * mean_q_t)).sum(1, keepdim=True)
        width_log_t = raw_log_t - 0.75 / second_moment - mean_raw_t / mean_raw_width
        return weights, widths, weights_t, width_log_t, beta, beta_t

    def component_bundle(self, local, time):
        """w, log(a), w_t, log(a)_t, beta, beta_t versus local fitting inputs."""
        n = self.time_coefficients.shape[0]
        w, a, wt, lt, beta, beta_t = self.components_from_raw(
            time.reshape(1), local[:n].reshape(1, n), local[n:2*n].reshape(1, n),
            local[2*n:])
        return torch.cat((w, a.log(), wt, lt, beta, beta_t), dim=1).reshape(-1)

    def value_component_bundle(self, local, time):
        n = self.time_coefficients.shape[0]
        w, a, beta = self.components_from_raw(
            time.reshape(1), local[:n].reshape(1, n), shape=local[n:])
        return torch.cat((w, a.log(), beta), dim=1).reshape(-1)

    @staticmethod
    def pack_shapes(value):
        """[...component,mode] -> [...mode*component], matching beta order."""
        return value.transpose(-1, -2).flatten(-2)

    @staticmethod
    def weighted_laguerre(q, order, derivatives=False):
        """exp(-q/2) times L_n and its polynomial derivatives (not product jets).

        Horner starts with the exponential, so every intermediate is finite
        even when the unweighted polynomial would overflow remotely.
        """
        exp = torch.exp(-q/2)
        coefficients = [(-1)**k*math.comb(order,k)/math.factorial(k) for k in range(order+1)]
        jets = []
        for _ in range(4 if derivatives else 1):
            value = torch.zeros_like(q)
            for coefficient in reversed(coefficients):
                value = value*q + exp*coefficient
            jets.append(value)
            coefficients = [(k+1)*coefficients[k+1] for k in range(len(coefficients)-1)]
        return jets if derivatives else jets[0]

    @staticmethod
    def weighted_tail(q, coefficients):
        exp = torch.exp(-q)
        value = torch.zeros_like(q)
        for coefficient in coefficients.flip(0):
            value = value*q + exp*coefficient
        return value*q

    def forward(self, points):
        x = points[:, :1].to(dtype=DTYPE).square()
        weights, widths, beta = self.temporal_components(points[:, 1])
        b4, b8 = beta.split(self.n_core+1, dim=1)
        q = x/widths
        shape = (torch.exp(-q/2) + b4*self.weighted_laguerre(q,4)
                 + b8*self.weighted_laguerre(q,8))
        return 9.5*(weights/widths*shape.square()/(1+b4.square()+b8.square())).sum(1,keepdim=True)

    def kernel_bundle(self, radii, widths, beta, *, sensitivities=False):
        """Kernel, q/shape jets and partial mass; parameter AD stays intact."""
        inv = widths[:,None,:].reciprocal()
        q = radii[None,:,None].square()*inv
        b = beta.reshape(-1,2,self.n_core+1).transpose(1,2)[:,None]
        b4,b8 = b[...,0],b[...,1]
        z = 1+b.square().sum(-1)
        l4,l8 = [self.weighted_laguerre(q,n,True) for n in (4,8)]
        L,Lq,Lqq,Lqqq = [torch.stack((l4[k],l8[k]),dim=-1) for k in range(4)]
        S = torch.exp(-q/2)+(b*L).sum(-1)
        Sq,Sqq,Sqqq = [(b*jet).sum(-1) for jet in (Lq,Lqq,Lqqq)]
        N,Nq,Nqq = S.square(),2*S*Sq,2*(Sq.square()+S*Sqq)
        f = 9.5*inv
        K,Kq,Kqq = f*N/z,f*(Nq-N)/z,f*(Nqq-2*Nq+N)/z
        Nb = 2*S[...,None]*L
        gb = Nb/z[...,None]-2*b*N[...,None]/z[...,None].square()
        Kb = f[...,None]*gb
        tl4,ts4,tl8,tc,ts8 = [self.weighted_tail(q,coeff) for coeff in
                              (self.tail_laguerre,self.tail_square,self.tail_laguerre8,
                               self.tail_cross,self.tail_square8)]
        tail = 2*b4*tl4+b4.square()*ts4+2*b8*tl8+2*b4*b8*tc+b8.square()*ts8
        H = -torch.expm1(-q)-tail/z
        if not sensitivities:
            return inv,q,K,Kq,Kqq,Kb,H
        Nqqq = 2*(3*Sq*Sqq+S*Sqqq)
        Nqb = 2*(L*Sq[...,None]+S[...,None]*Lq)
        Nqqb = 2*(2*Sq[...,None]*Lq+L*Sqq[...,None]+S[...,None]*Lqq)
        gqb = Nqb/z[...,None]-2*b*Nq[...,None]/z[...,None].square()
        gqqb = Nqqb/z[...,None]-2*b*Nqq[...,None]/z[...,None].square()
        # Directional beta_t contraction of the shape Hessian is computed
        # later; its small 2x2 blocks have no radial/time derivative shortcut.
        eye = torch.eye(2,dtype=q.dtype,device=q.device)
        zz=z[...,None,None]
        gbb = (2*L[..., :,None]*L[...,None,:]/zz
               -4*S[...,None,None]*(L[..., :,None]*b[...,None,:]+b[..., :,None]*L[...,None,:])/zz.square()
               -2*N[...,None,None]*eye/zz.square()
               +8*N[...,None,None]*b[..., :,None]*b[...,None,:]/zz.pow(3))
        Kqqq=f*(Nqqq-3*Nqq+3*Nq-N)/z
        Kqb=f[...,None]*(gqb-gb)
        Kqqb=f[...,None]*(gqqb-2*gqb+gb)
        Kbb=f[...,None,None]*gbb
        tail_b=torch.stack((2*tl4+2*b4*ts4+2*b8*tc,
                            2*tl8+2*b4*tc+2*b8*ts8),dim=-1)
        Hb=-tail_b/z[...,None]+2*b*tail[...,None]/z[...,None].square()
        return inv,q,K,Kq,Kqq,Kb,H,Kqqq,Kqb,Kqqb,Kbb,Hb

    def differential_parts(self, radii, times):
        w,a,wt,lt,beta,beta_t = self.temporal_components(times,True)
        inv,q,K,Kq,Kqq,Kb,H = self.kernel_bundle(radii,a,beta)
        w,wt,lt = [v[:,None,:] for v in (w,wt,lt)]
        bt=beta_t.reshape(-1,2,self.n_core+1).transpose(1,2)[:,None]
        u=(w*K).sum(2)
        ut=(wt*K+w*lt*(-K-q*Kq)+w*(bt*Kb).sum(-1)).sum(2)
        ux=(w*inv*Kq).sum(2)
        mass=4.75*(w*H).sum(2)
        diffusion=4*(w*inv*(q*Kqq+Kq)).sum(2)
        chemistry=2*mass*ux+u.square()
        return u,ut,diffusion,chemistry,mass

    def residual_sensitivities(self, radii, times, components=None):
        """Exact local dR/d[w,log(a),w_t,log(a)_t,beta,beta_t]."""
        if components is None:
            components=self.temporal_components(times,True)
        weights,widths,weights_t,log_t,beta,beta_t=components
        w,wt,lt=[v[:,None,:] for v in (weights,weights_t,log_t)]
        bt=beta_t.reshape(-1,2,self.n_core+1).transpose(1,2)[:,None]
        inv,q,K,kq,kqq,kb,H,kqqq,kqb,kqqb,kbb,Hb=self.kernel_bundle(radii,widths,beta,sensitivities=True)
        ka=-K-q*kq
        kaa=K+3*q*kq+q.square()*kqq
        kab=-kb-q[...,None]*kqb
        u=(w*K).sum(2,keepdim=True)
        ux=(w*inv*kq).sum(2,keepdim=True)
        mass=4.75*(w*H).sum(2,keepdim=True)
        ut=(wt*K+w*lt*ka+w*(bt*kb).sum(-1)).sum(2,keepdim=True)
        diffusion=4*(w*inv*(q*kqq+kq)).sum(2,keepdim=True)
        residual=ut-diffusion-2*mass*ux-u.square()
        rw=(lt*ka+(bt*kb).sum(-1)-4*inv*(q*kqq+kq)
            -2*(4.75*H*ux+mass*inv*kq)-2*u*K)
        ma=-.5*w*q*K/inv
        uxa=-w*inv*(2*kq+q*kqq)
        uta=wt*ka+w*lt*kaa+w*(bt*kab).sum(-1)
        da=-4*w*inv*(q.square()*kqqq+4*q*kqq+2*kq)
        ra=uta-da-2*(mass*uxa+ux*ma)-2*u*w*ka
        rb=(wt[...,None]*kb+w[...,None]*lt[...,None]*kab
            +w[...,None]*(kbb*bt[...,None,:]).sum(-1)
            -4*(w*inv)[...,None]*(q[...,None]*kqqb+kqb)
            -2*(4.75*w[...,None]*Hb*ux[...,None]+mass[...,None]*(w*inv)[...,None]*kqb)
            -2*u[...,None]*w[...,None]*kb)
        return residual[:,:,0],torch.cat((rw,ra,K,w*ka,self.pack_shapes(rb),
                                         self.pack_shapes(w[...,None]*kb)),dim=2)
