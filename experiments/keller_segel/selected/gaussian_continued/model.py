"""Warm Gaussian mixture with bounded corrections in physical curve coordinates.

Scientific hypothesis: the inherited second-jet family is useful, but fitting
its polynomial inputs through 12*tanh(P/12) weakens some temporal directions.
Retain its fitted field exactly at zero correction and fit

    C(t) = 12*tanh(P_inherited(t)/12) + b*tanh(P_delta(t)/b), b=6.

Corrections enter AFTER the inherited saturation. At P_delta=0 their derivative
is one even where the inherited input map was nearly flat. This changes the
bounded temporal family, not the radial Gaussian kernels or the raw residual
objective. It is a warm continuation, not an independently discovered solution.
The amplitude bound permits excursions beyond the old [-12,12] curve range;
it does not prescribe a concentration time or a target trajectory.

The initial baseline is solver_acff6d6d0001's public fitted state. The 312
inherited temporal coefficients below are registered as a persistent buffer;
twelve fixed log rates are the same as that source. No checkpoint, optimizer,
loss module or example file is read during construction or cold prediction.
Every field-defining fitted quantity is exported in state_dict. Default counts:
312 trainable correction coefficients; 325 fixed buffer scalars (312 inherited
fitted coefficients, 12 log rates, 1 correction bound). Setup allocates these
small arrays and does no fitting or differentiation.

Inherited evidence, not certificates: same first-jet class with L-BFGS had
public physics loss 1.7049e-4; Gauss--Newton had 1.9539e-6. Increasing temporal
degree alone gave 4.1896e-6. The second-jet source had 5.2785e-8 public physics
and 6.4910e-8 best training loss after 1839 steps. Its history remained improving
at the allowance boundary. On 201 uniformly spaced synthetic times its old
squash derivative ranged from 1.1946e-4 to one; only 0.166% were below .01.
Thus saturation is localized, and may not explain the remaining local defects.
Synthetic validation (ASYS_NUM_THREADS=2; no optimizer steps): the fixed
checker passes, including parameter sensitivity. Nonzero .02*sin(i) correction
coefficients preserve initial value/tangent/second jet, mass/moment and symmetry.
Analytic PDE terms agree with ordinary coordinate AD to about 1.5e-10 absolute;
the factored PDE Jacobian directional error is 1.1e-11, and J^T R agrees with
the loss gradient to 8.6e-14. A strict state export with synthetic nonzero
corrections reproduces all protected cold field probes in a fresh Python
process with zero difference, without importing candidate training code.
These are algebra checks, not convergence evidence.
The zero-correction field agrees exactly with its inherited field on a 94-radius by
47-time synthetic grid, and its first derivatives agree to 1.9e-12 absolute.
The source's worst sampled raw residuals occur both early and at the horizon;
neither low residuals nor exact global moments certify the IVP trajectory.

Measured model construction/import was .009 s and optimizer construction
1.68 s, with no fitting in either setup. Three full configured synthetic
loss/backward calls took .029-.040 s; normal matrices took .103-.179 s.
Those samples establish feasibility only, not whole-fit convergence or query
latency. Zero corrections give training loss 6.43384e-8, slightly different
from the source because regularization now penalizes correction coefficients.
There is no fitted convergence claim for these new coordinates.

u=9.5*sum_j w_j/a_j*exp(-r^2/a_j), with w_j>=0, a_j>0,
sum w_j=1 and sum w_j*a_j=1-.75*t. Radial mass is 4.75 and the
whole-space second radial moment is 4.75-3.5625*t. Finite-radius flux and
remote decay must still be checked. The initial u, u_t and u_tt are exact for
all corrections: two positive core groups enter at t and t^2 with widths
1/2 and 1/3, using the PDE-derived jets in the inherited construction.
Ordinary forward uses PyTorch coordinate AD; training shortcuts are analytic
identities for this same field, with intact parameter gradients.

Live primary literature consulted:
Blanchet, Dolbeault & Perthame (2006), Lemma 3: mass/moment identities;
https://www.ceremade.dauphine.fr/~dolbeaul/Preprints/Fichiers/Keller-Segel-30.pdf
Davila et al. (2024), Eq. (1.2): critical stationary profile 8/(1+r^2)^2;
https://link.springer.com/article/10.1007/s00205-024-02006-7
That profile is a positive Gaussian scale mixture, motivating the inherited
kernels, not supplying this finite-window Gaussian-IVP trajectory.
Rathore et al. (ICML 2024): differential-residual losses can be ill-conditioned;
https://proceedings.mlr.press/v235/rathore24a.html
The bounded correction here is our conditioning hypothesis, not their NNCG.
"""
from __future__ import annotations

import math

import torch
from torch import nn

DTYPE = torch.float64
T_MAX = 0.86652

# Exact scalar export from solver_examples/solver_acff6d6d0001/logs/evaluate/eval_artifacts/state_dict.pt.
# Source checkpoint SHA256: 59a9b71361775e1e44ee31894fbfdc6b16dd70be43d10790a3a4ab2697f454d2
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

# Active fitted CFC initialization from cfc_start.yaml; retained state SHA256: 0cd6275d82e98cc335edcbe52735e130354f31287514a00697f05bd724793dbd
_CFC_START_TIME_COEFFICIENTS = [
    [-0.19987257602211178, 1.4926896038070476, -3.6546806016414135, 4.5002570678986915, -3.106323680762235, 0.7615571263817821, 0.972141256355076, -1.4536952359716542, 1.1165181547010674, -0.6269040961306289, 0.27895339812389114, -0.100483349534605, 0.021047302129698166],
    [0.18872726906987586, -0.3456652985008329, 0.15093552531943655, 0.04887414349182234, -0.08083781202730453, -0.006938680308684923, 0.15702221148871331, -0.16987191176380392, 0.10446411658138233, -0.08295612401613241, 0.06837987620285485, -0.04362424918820454, 0.013054581621305563],
    [0.755968557198494, -1.7436767503495314, 2.509304624113949, -3.4470500301451525, 3.8950744255642404, -3.4055250809140745, 2.130659576543306, -0.925437941741556, 0.22464525988779674, 0.08347265365547199, -0.11807923537923612, 0.05493420608705472, -0.013937073375588872],
    [-0.9588118035030909, 2.206225637644175, -2.750684712085273, 3.065664716080803, -2.940828332318601, 2.4732889987408315, -1.9431526248818942, 1.4635775022810469, -1.0324603120902576, 0.6554583285372012, -0.3452440639077618, 0.1319102899803153, -0.025764566782561595],
    [-0.9619505568281721, 2.5011464133903702, -3.501387806582047, 3.825890593330224, -3.1229227076340473, 2.01096708668457, -1.169128876146343, 0.5150290448645345, -0.05657533327900495, -0.08412850261152327, 0.0712850499533327, -0.03420060350868381, 0.005787456870643625],
    [1.0628392493227743, -1.8541280295123175, 0.5997560501314783, 1.8592614363393611, -4.465625429652804, 5.82809266483052, -5.4569263930401535, 3.915440867535415, -2.190627184992312, 0.9565238491223559, -0.317273289225415, 0.0747753751542312, -0.011831238374196727],
    [0.072701833437271, -0.9432631464846674, 1.8679572151900306, -1.5234659940610646, 1.0331788561511532, -1.3508711300027063, 1.8234696459433726, -1.7115901944626766, 1.0432959651551001, -0.3701548095568006, 0.07024582936491797, -0.016477480588487005, 0.004659658713035749],
    [-0.3632747709509824, 0.36796765949372773, 0.24297505736760242, -0.2039266029038357, -0.5927425956754139, 1.0852479899953646, -0.3470371759852842, -1.1278780434559614, 2.032971755447793, -1.8102094017020771, 0.9925146613119212, -0.3323520458372227, 0.05447086016760526],
    [1.4749065011118507, -2.700500799051016, 2.134688775834759, -1.399039274872213, 0.6186423937653627, 0.03420483951580335, -0.5111372148473541, 0.6943471891756503, -0.6387064808547174, 0.4900465429891922, -0.2912066507703311, 0.118679137643215, -0.025959638800987787],
    [-0.9239591263606162, 1.8074262321165573, -1.7137436036976776, 1.5616099808012855, -1.3493685393188704, 1.1633340996232377, -1.0170020540320495, 0.8327574244571051, -0.5858963513544935, 0.338571325067593, -0.15089147386707644, 0.04599031504159018, -0.0085816520029111],
    [-1.5483637365583944, 2.993616986224412, -2.754201277838509, 2.3974351329992305, -1.8900890082302326, 1.4085353522129527, -1.0792428093028241, 0.7895699852659046, -0.48639014429127664, 0.24070191927363357, -0.09057274540476001, 0.024915552857111282, -0.005686807993445181],
    [-2.1790339342146816, 4.2205450467636325, -3.8498740368381883, 3.2282157515666405, -2.39123677039388, 1.6606221542547386, -1.1804245663624566, 0.7824148808885941, -0.436370404151396, 0.2013066511718118, -0.06372816222410507, 0.008829110426430233, -0.0012369373813072406],
    [-0.03340596743094543, 0.08603769033288726, -0.09508522331337182, -0.04466966096263735, 0.30584812209242024, -0.3887043743516482, 0.27408131077004705, -0.18538704401768394, 0.1317998536989548, -0.08907400645200668, 0.07470178144022341, -0.05042726459893974, 0.01349901038457548],
    [-1.5090607576260882, 3.028133532667493, -2.9327005320675474, 2.6007209221613286, -2.0837815108369178, 1.6011591634414115, -1.291691561862129, 1.0228591390103006, -0.6790047632765827, 0.37306231416855623, -0.1901848474978422, 0.06562350114450408, -0.006834483989614904],
    [0.1892149701793046, -0.6635484674974159, 1.165992853878931, -1.3401815454383375, 1.054365790074435, -0.5431620108652196, 0.19576917058561988, -0.17847399732787772, 0.27295227604825817, -0.2527777857895399, 0.1464942747412217, -0.05758465083507604, 0.012594799539127378],
    [-1.2045894068500416, 2.031492104428562, -1.255032413442827, 0.8644763871099345, -1.1031187648668743, 1.538513238527393, -1.6548899838118303, 1.199925182849192, -0.5707702987744039, 0.2047171362085755, -0.0620661286644471, 0.014579558111503108, -0.003880097671645289],
    [-0.9235563092332975, 1.3906172134323755, -0.5510150388030897, 0.17665575503455785, -0.5181377392240791, 1.136553242728416, -1.4069663696534824, 1.1483802075937162, -0.6754249074875294, 0.3220534327908532, -0.14153300632174134, 0.049368701068863305, -0.007715703182221404],
    [-1.585554724181491, 1.8270467661873926, 0.562981297164084, -1.9089938015059105, 2.053736703709428, -2.0260234880017793, 2.408720921726948, -2.3756179122891528, 1.4688697786628375, -0.4310399798988847, -0.058905210136734484, 0.0867484609160547, -0.019847595112321995],
    [-1.2251726644259517, 2.651210182163446, -2.461146279836965, 0.6439611401281604, 1.5279362192700596, -3.115513276268112, 5.24036336868789, -6.8922682579991035, 6.618553519848615, -4.63082470129494, 2.17471348481466, -0.638558152638755, 0.10590079929777889],
    [0.09653951540465952, -0.04411412745387876, -0.15248149159920107, 0.16698316789522977, 0.0509917507354381, -0.2719514302555415, 0.3366628802901646, -0.44630079257732874, 0.43543077040855444, -0.2380251027003889, 0.11813329769732367, -0.07023414445885637, 0.018323469357867373],
    [0.6610928419224608, -1.1648770783212894, 0.9900429770477256, -0.9377695095626238, 0.8758843207008585, -0.7210486758239608, 0.5195842850626986, -0.45273940914752536, 0.3412591261781676, -0.15880897587207188, 0.09184306146553753, -0.07208858444703727, 0.02584061356650168],
    [0.9873550710596196, -1.9365677167846722, 1.8075014495891788, -1.5544114061503043, 1.2830299724726797, -1.0432687678129262, 0.7811273326754464, -0.5501536376547542, 0.3415865851697491, -0.1778352206650054, 0.10600167444266456, -0.061505922957498155, 0.016294485931021785],
    [0.015409491360264574, -0.49280324942024156, 0.661481085954254, -0.3263103101493582, 0.5235904005490177, -0.6603977584214414, 0.42740678811553723, -0.31630710793242933, 0.31862710930486193, -0.20776628331123068, 0.08836995221783384, -0.05199447057686786, 0.02093385758903208],
    [-0.004683049332665294, -0.1467992701293023, 0.19085712862082974, -0.29163560369174996, 0.25148671631525116, -0.11228465660615189, 0.334475995005036, -0.10692153389452366, 0.051017583787807, -0.27413171728031654, 0.07090824673592619, -0.047106077328229475, 0.076808389886254],
]


class Candidate(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        self.n_core = int(config.get("n_core", 12))
        self.n_first = int(config.get("n_first", self.n_core // 2))
        if not 1 <= self.n_first < self.n_core:
            raise ValueError("Both positive core groups must have at least one component")
        self.degree = int(config.get("time_degree", 12))
        if self.n_core != 12 or self.n_first != 6:
            raise ValueError("The disclosed inherited baseline has two groups of six")
        self.seed_degree = 12
        self.register_buffer("inherited_time_coefficients", torch.tensor(
            _INHERITED_TIME_COEFFICIENTS, dtype=DTYPE))
        bound = float(config.get("correction_bound", 6.0))
        if not 0 < bound <= 12:
            raise ValueError("correction_bound must be in (0,12]")
        self.register_buffer("correction_bound", torch.tensor(bound, dtype=DTYPE))
        # Two group masses, K widths, (K-2) conditional logits.
        self.time_coefficients = nn.Parameter(
            torch.tensor(_CFC_START_TIME_COEFFICIENTS, dtype=DTYPE)
        )
        self.register_buffer(
            "log_rates",
            torch.cat([
                torch.linspace(math.log(0.04), math.log(10.0), count, dtype=DTYPE)
                for count in (self.n_first, self.n_core - self.n_first)
            ]),
        )

    def time_basis(self, tau, derivatives=False):
        """Trainable correction basis and its derivative with respect to tau."""
        return self._chebyshev(tau, self.degree, derivatives)

    @staticmethod
    def _chebyshev(tau, degree, derivatives=False):
        """Fixed polynomial evaluation; independent of fitted state."""
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
            return self.components_from_raw(times, raw, raw_tau)
        return self.components_from_raw(times, self.raw_curves(times))

    def components_from_raw(self, times, raw, raw_tau=None):
        """Same temporal algebra with explicit polynomial values/slopes.

        This separation permits a small exact training Jacobian. Forward
        values and coordinate derivatives always use ordinary PyTorch AD.
        """
        t = times.reshape(-1, 1).to(dtype=DTYPE)
        tau = t / T_MAX
        # Frozen inherited curve and its TRUE time derivative remain in the
        # ordinary graph. They are not detached values or a time lookup.
        if raw_tau is None:
            seed_basis = self._chebyshev(tau, self.seed_degree)
            seed_raw = seed_basis @ self.inherited_time_coefficients.T
        else:
            seed_basis, seed_basis_tau = self._chebyshev(tau, self.seed_degree, True)
            seed_raw = seed_basis @ self.inherited_time_coefficients.T
            seed_raw_tau = seed_basis_tau @ self.inherited_time_coefficients.T
        seed_squashed = torch.tanh(seed_raw / 12.0)
        delta_squashed = torch.tanh(raw / self.correction_bound)
        correction = 12.0 * seed_squashed + self.correction_bound * delta_squashed
        if raw_tau is not None:
            correction_tau = ((1 - seed_squashed.square()) * seed_raw_tau
                              + (1 - delta_squashed.square()) * raw_tau)
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
        mean_raw_width = (weights * raw_widths).sum(1, keepdim=True)
        second_moment = 1 - 0.75 * t
        widths = raw_widths * second_moment / mean_raw_width
        if raw_tau is None:
            return weights, widths

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
        mean_raw_t = (raw_widths * (weights_t + weights * raw_log_t)).sum(1, keepdim=True)
        width_log_t = raw_log_t - 0.75 / second_moment - mean_raw_t / mean_raw_width
        return weights, widths, weights_t, width_log_t

    def component_bundle(self, local, time):
        """w, log(a), w_t, (log(a))_t at one time, for numerical fitting."""
        n = self.time_coefficients.shape[0]
        w, a, wt, lt = self.components_from_raw(
            time.reshape(1), local[:n].reshape(1, n), local[n:].reshape(1, n)
        )
        return torch.cat((w, a.log(), wt, lt), dim=1).reshape(-1)

    def value_component_bundle(self, local, time):
        """w and log(a) only; boundary values do not need time slopes."""
        w, a = self.components_from_raw(time.reshape(1), local.reshape(1, -1))
        return torch.cat((w, a.log()), dim=1).reshape(-1)

    def forward(self, points):
        x = points[:, :1].to(dtype=DTYPE).square()
        weights, widths = self.temporal_components(points[:, 1])
        return 9.5 * (weights / widths * torch.exp(-x / widths)).sum(1, keepdim=True)

    def differential_parts(self, radii, times):
        """Exact continuous u, u_t, diffusion, chemotaxis and partial mass.

        Using x=r^2 gives D=4(x*u_xx+u_x), C=2*M*u_x+u^2.
        Mass terms retain parameter gradients.
        """
        weights, widths, weights_t, width_log_t = self.temporal_components(times, True)
        x = radii.reshape(1, -1, 1).square()
        inv_a = widths[:, None, :].reciprocal()
        q = x * inv_a
        kernel = 9.5 * inv_a * torch.exp(-q)
        w = weights[:, None, :]
        u = (w * kernel).sum(2)
        ut = (kernel * (weights_t[:, None, :] +
                       w * width_log_t[:, None, :] * (q - 1))).sum(2)
        ux = -(w * kernel * inv_a).sum(2)
        uxx = (w * kernel * inv_a.square()).sum(2)
        mass = 4.75 * (w * (-torch.expm1(-q))).sum(2)
        diffusion = 4 * (x[:, :, 0] * uxx + ux)
        chemistry = 2 * mass * ux + u.square()
        return u, ut, diffusion, chemistry, mass
