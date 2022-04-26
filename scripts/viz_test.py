import numpy as np
import math
import matplotlib.pyplot as plt
ranges = [4.561761379241943, 4.535043239593506, 4.507310390472412, 4.474637985229492, 4.44512939453125, 4.420141220092773, 4.393043518066406, 4.364387512207031, 4.337436676025391, 4.31394624710083, 4.288874626159668, 4.262103080749512, 4.238335609436035, 4.211997032165527, 4.191660404205322, 4.167182922363281, 4.1442742347717285, 4.123353958129883, 4.104182720184326, 4.083096504211426, 4.0603928565979, 4.040062427520752, 4.019944667816162, 3.996957540512085, 3.981504440307617, 3.9577226638793945, 3.944021463394165, 3.9217498302459717, 3.906580924987793, 3.8885276317596436, 3.8733012676239014, 3.8508846759796143, 3.834282159805298, 3.8196775913238525, 3.80181884765625, 3.78721284866333, 3.7731850147247314, 3.758410930633545, 3.7419302463531494, 3.725623607635498, 3.7165942192077637, 3.699793815612793, 3.6877009868621826, 3.671402931213379, 3.6554203033447266, 3.6455953121185303, 3.634049892425537, 3.6194021701812744, 3.607468843460083, 3.5954806804656982, 3.581735849380493, 3.569260597229004, 3.560783863067627, 3.552393674850464, 3.538602352142334, 3.5286245346069336, 3.513002395629883, 3.5038509368896484, 3.496771812438965, 3.483224391937256, 3.4745771884918213, 3.4695072174072266, 3.459862232208252, 3.4518654346466064, 3.4425573348999023, 3.4313788414001465, 3.4205856323242188, 3.4121899604797363, 3.407043218612671, 3.3985087871551514, 3.389451265335083, 3.3802618980407715, 3.3765203952789307, 3.3682103157043457, 3.359938859939575, 3.353564977645874, 3.3453874588012695, 3.33721923828125, 3.3350603580474854, 3.327056884765625, 3.32254695892334, 3.3154027462005615, 3.31192946434021, 3.3051469326019287, 3.2995126247406006, 3.292649984359741, 3.289198398590088, 3.283383369445801, 3.28017258644104, 3.2709150314331055, 3.2684106826782227, 3.2671074867248535, 3.260155439376831, 3.2578275203704834, 3.2517783641815186, 3.248382568359375, 3.2470409870147705, 3.2401537895202637, 3.238401412963867, 3.2368874549865723, 3.2311625480651855, 3.229254961013794, 3.225579261779785, 3.225105047225952, 3.221156120300293, 3.2187376022338867, 3.219172477722168, 3.218759059906006, 3.214149236679077, 3.212413787841797, 3.2103466987609863, 3.207986831665039, 3.2069525718688965, 3.2070586681365967, 3.2048263549804688, 3.2043333053588867, 3.2046923637390137, 3.2027931213378906, 3.201115608215332, 3.202103614807129, 3.2001538276672363, 3.2018940448760986, 3.2042129039764404, 3.2016537189483643, 3.202040195465088, 3.204174757003784, 3.203826904296875, 3.2055413722991943, 3.2061104774475098, 3.2062041759490967, 3.2054760456085205, 3.2088611125946045, 3.2091310024261475, 3.2117278575897217, 3.2126030921936035, 3.214655876159668, 3.2182135581970215, 3.220959424972534, 3.222155809402466, 3.2258095741271973, 3.227642059326172, 3.230642080307007, 3.2349274158477783, 3.237232208251953, 3.2391812801361084, 3.241593837738037, 3.245333433151245, 3.2495651245117188, 3.254164218902588, 3.256035566329956, 3.262427568435669, 3.264655351638794, 3.2688233852386475, 3.2735047340393066, 3.280320405960083, 3.284100294113159, 3.2869231700897217, 3.294149398803711, 3.298933267593384, 3.3033039569854736, 3.3086397647857666, 3.3145482540130615, 3.3202078342437744, 3.32580304145813, 3.3341429233551025, 3.3390955924987793, 3.3468852043151855, 3.353297233581543, 3.359886646270752, 3.3696775436401367, 3.375916004180908, 3.3840606212615967, 3.3912031650543213, 3.3984806537628174, 3.4079997539520264, 3.4145493507385254, 3.4242429733276367, 3.4332315921783447, 3.441697597503662, 3.4508607387542725, 3.4588074684143066, 3.470350503921509, 3.4803295135498047, 3.4904963970184326, 3.499066114425659, 3.509737014770508, 3.5230724811553955, 3.5309414863586426, 3.542567491531372, 3.554940938949585, 3.565462112426758, 3.5769243240356445, 3.575615167617798, 3.5896759033203125, 3.6012206077575684, 3.6142055988311768, 3.625904083251953, 3.6402337551116943, 3.6544148921966553, 3.665574073791504, 3.679128646850586, 3.693605899810791, 3.7084174156188965, 3.721439838409424, 3.7369186878204346, 3.7503955364227295, 3.7662854194641113, 3.781599283218384, 3.7986104488372803, 3.811727523803711, 3.8300111293792725, 3.843601703643799, 3.861314535140991, 3.8976426124572754, 3.913243532180786, 3.931110382080078, 3.949378728866577, 3.966088056564331, 3.9866504669189453, 4.003768444061279, 4.023112773895264, 4.0426249504089355, 4.061501979827881, 4.08223295211792, 4.102382183074951, 4.122523307800293, 4.142825603485107, 4.164356231689453, 4.186384201049805, 4.205310344696045, 4.227364540100098, 4.273439884185791, 4.294895172119141, 4.3194708824157715, 4.342105865478516, 4.366080284118652, 4.389290809631348, 4.412450790405273, 4.437155723571777, 4.486924648284912, 4.512407302856445, 4.5379157066345215, 4.564865589141846, 4.5906829833984375, 4.6173014640808105, 4.644372463226318, 4.700133323669434, 4.726479530334473, 4.757246971130371, 4.784008502960205, 4.814125061035156, 4.873313903808594, 4.903111934661865, 4.932809829711914, 4.963759422302246, 5.025445461273193, 5.058828353881836, 5.089395523071289, 5.122412204742432, 5.190220355987549, 5.222692489624023, 5.255504608154297, 5.323142051696777, 5.361469268798828, 5.396800994873047, 5.465672969818115, 5.503199577331543, 5.541057109832764, 5.616791725158691, 5.656525135040283, 5.733129501342773, 5.771446704864502, 5.851747035980225, 5.892458438873291, 5.935217380523682, 6.018843173980713, 6.061543941497803, 6.145938873291016, 6.188546657562256, 6.277178764343262, 6.3712639808654785, 6.416483402252197, 6.508944988250732, 6.55684757232666, 6.654013156890869, 6.750210285186768, 6.805639266967773, 6.905484676361084, 7.0093560218811035, 7.065070629119873, 7.169379711151123, 7.278507709503174, 7.334285259246826, 7.446892738342285, 7.562443733215332, 7.680640697479248, 7.801360130310059, 7.923730850219727, 8.048487663269043, 8.112360000610352, 8.241374969482422, 8.375048637390137, 8.508350372314453, 8.646232604980469, 8.86022663116455, 9.004125595092773, 9.154804229736328, 9.304277420043945, 9.46080207824707, 9.618754386901855, 9.865784645080566, 10.031225204467773, 10.204030990600586, 10.380376815795898, 10.649299621582031, 10.83596420288086, 11.123113632202148, 11.320711135864258, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 1.9396082162857056, 1.9294676780700684, 1.906692624092102, 1.9042308330535889, 1.903833270072937, 1.9037762880325317, 1.9032126665115356, 1.905641794204712, 1.9109902381896973, 1.921272873878479, 27.333106994628906, 27.310359954833984, 27.29865074157715, 27.28047752380371, 27.267871856689453, 27.25470542907715, 27.24235725402832, 27.22941780090332, 27.219091415405273, 27.212785720825195, 27.20602798461914, 27.19997215270996, 27.19373893737793, 27.19074249267578, 27.186960220336914, 27.18649673461914, 27.18743324279785, 27.15842056274414, 27.36363983154297, 27.54790496826172, 27.4467830657959, 27.397226333618164, 27.55393409729004, 27.314664840698242, 27.514707565307617, 27.712385177612305, 26.782865524291992, 24.356239318847656, 22.675373077392578, 20.83255958557129, 19.529422760009766, 18.08368682861328, 17.048583984375, 16.107006072998047, 15.244878768920898, 14.454684257507324, 1.779221534729004, 1.7661446332931519, 1.7618759870529175, 1.7586179971694946, 1.7569881677627563, 1.757096290588379, 1.7632397413253784, 1.770368218421936, 1.7825416326522827, 1.7911269664764404, 9.194019317626953, 8.914185523986816, 8.643389701843262, 8.387331008911133, 8.142200469970703, 7.903988838195801, 7.678035259246826, 7.460509777069092, 7.319533348083496, 7.1158318519592285, 6.918362140655518, 6.7921462059021, 6.607730388641357, 6.489342212677002, 6.314623832702637, 6.203239440917969, 6.041106700897217, 5.936329364776611, 5.830747127532959, 5.733205795288086, 5.584883213043213, 5.491695880889893, 5.398939609527588, 5.306888103485107, 5.217828750610352, 5.1317033767700195, 5.007633209228516, 4.9226508140563965, 4.8437724113464355, 4.764505863189697, 4.688344955444336, 4.650147438049316, 4.575923919677734, 4.502196311950684, 4.432750701904297, 4.3631792068481445, 4.29443359375, 4.228054046630859, 4.193538665771484, 4.130250453948975, 4.067002296447754, 4.034977436065674, 3.974163770675659, 3.915156841278076, 3.8855700492858887, 3.8268909454345703, 3.7729554176330566, 3.744361400604248, 3.6872806549072266, 3.6585066318511963, 3.607536792755127, 3.5546927452087402, 3.529148578643799, 3.4779820442199707, 3.45292329788208, 3.405430316925049, 3.3790793418884277, 3.3578779697418213, 3.309392213821411, 3.287196397781372, 3.2397420406341553, 3.2187557220458984, 3.1742777824401855, 3.1533730030059814, 3.1326231956481934, 3.0900392532348633, 3.0693447589874268, 3.046298027038574, 3.006767511367798, 2.987274646759033, 2.9683845043182373, 2.9301342964172363, 2.912400960922241, 2.8937923908233643, 2.8749747276306152, 2.839160203933716, 2.819049119949341, 2.8035385608673096, 2.7859957218170166, 2.749744176864624, 2.732919692993164, 2.717191696166992, 2.700223684310913, 2.684842824935913, 2.653414487838745, 2.636436700820923, 2.621476888656616, 2.6059553623199463, 2.5906929969787598, 2.5747878551483154, 2.5608608722686768, 2.533039093017578, 2.517367362976074, 2.5035006999969482, 2.490104913711548, 2.475850820541382, 2.4622669219970703, 2.44804310798645, 2.4356331825256348, 2.4085826873779297, 2.398300886154175, 2.3850646018981934, 2.373063564300537, 2.359891653060913, 2.347876787185669, 2.337106227874756, 2.326364517211914, 2.3148205280303955, 2.3021018505096436, 2.292173385620117, 2.280527114868164, 2.2690398693084717, 2.256873369216919, 2.2479310035705566, 2.2370898723602295, 2.2258172035217285, 2.2164318561553955, 2.1987996101379395, 2.1886367797851562, 2.176497459411621, 2.168374538421631, 2.1598031520843506, 2.1518077850341797, 2.141834020614624, 2.1315994262695312, 2.1253230571746826, 2.1166975498199463, 2.1066360473632812, 2.098200798034668, 2.0916032791137695, 2.082415819168091, 2.075901746749878, 2.0674123764038086, 2.0610954761505127, 2.052623987197876, 2.045614242553711, 2.0387542247772217, 2.0309829711914062, 2.0311267375946045, 2.0258922576904297, 2.0181734561920166, 2.0111114978790283, 2.004612445831299, 2.000065565109253, 1.9930665493011475, 1.9871324300765991, 1.980529546737671, 1.9736593961715698, 1.9703534841537476, 1.962936282157898, 1.9596871137619019, 1.9546886682510376, 1.9468690156936646, 1.9422305822372437, 1.938169002532959, 1.9306817054748535, 1.928206443786621, 1.9222041368484497, 1.9197286367416382, 1.9136240482330322, 1.910866379737854, 1.90614914894104, 1.9021353721618652, 1.898468017578125, 1.8941242694854736, 1.8907392024993896, 1.8851091861724854, 1.881656289100647, 1.879729151725769, 1.8748503923416138, 1.872771978378296, 1.8686186075210571, 1.8665562868118286, 1.8629828691482544, 1.8607560396194458, 1.8573880195617676, 1.854948878288269, 1.8525861501693726, 1.8494247198104858, 1.8469685316085815, 1.8445556163787842, 1.8429800271987915, 1.83784019947052, 1.8374053239822388, 1.8371740579605103, 1.8357841968536377, 1.8311777114868164, 1.8308751583099365, 1.8298763036727905, 1.8272371292114258, 1.825217604637146, 1.8246426582336426, 1.8228932619094849, 1.8211238384246826, 1.8207857608795166, 1.8200621604919434, 1.8155490159988403, 1.8183242082595825, 1.815773367881775, 1.8153482675552368, 1.8137484788894653, 1.8132463693618774, 1.8134852647781372, 1.8126636743545532, 1.8108044862747192, 1.8116520643234253, 1.8110861778259277, 1.8102720975875854, 1.811284065246582, 1.8103656768798828, 1.8110361099243164, 1.8108396530151367, 1.8105437755584717, 1.8102706670761108, 1.8116499185562134, 1.8111680746078491, 1.8118171691894531, 1.8134782314300537, 1.8147941827774048, 1.8134573698043823, 1.8137961626052856, 1.8167253732681274, 1.8177945613861084, 1.8176745176315308, 1.8173837661743164, 1.8201018571853638, 1.8218717575073242, 1.823076605796814, 1.823907732963562, 1.8249305486679077, 1.8282930850982666, 1.827504277229309, 1.827856183052063, 1.83320152759552, 1.8332059383392334, 1.8354893922805786, 1.8372291326522827, 1.8401403427124023, 1.843796730041504, 1.8449944257736206, 1.845887303352356, 1.8510128259658813, 1.8533724546432495, 1.8578413724899292, 1.8551610708236694, 1.8610724210739136, 1.8633815050125122, 1.8688451051712036, 1.8680968284606934, 1.8743772506713867, 1.8757576942443848, 1.8811430931091309, 1.8832736015319824, 1.8856481313705444, 1.8902627229690552, 1.8942171335220337, 1.8969874382019043, 1.9031833410263062, 1.9047338962554932, 1.910182237625122, 1.9161378145217896, 1.9199239015579224, 1.9230952262878418, 1.9259599447250366, 1.9344207048416138, 1.93783700466156, 1.9448233842849731, 1.949875831604004, 1.9543819427490234, 1.957806944847107, 1.9612435102462769, 1.965811014175415, 1.9736005067825317, 1.9779406785964966, 1.9869790077209473, 1.9915456771850586, 1.999644160270691, 2.0029408931732178, 2.010580539703369, 2.0168519020080566, 2.023561477661133, 2.0298075675964355, 2.036442756652832, 2.0452075004577637, 2.0509047508239746, 2.0566246509552, 2.065316677093506, 2.074449300765991, 2.0811421871185303, 2.091252565383911, 2.095296621322632, 2.1031508445739746, 2.1131882667541504, 2.120046854019165, 2.1301772594451904, 2.1369426250457764, 2.1464157104492188, 2.1545889377593994, 2.1652770042419434, 2.177910327911377, 2.1860740184783936, 2.1947059631347656, 2.2044754028320312, 2.2163403034210205, 2.2243082523345947, 2.2350010871887207, 2.2446908950805664, 2.2568259239196777, 2.267080307006836, 2.28257417678833, 2.2918496131896973, 2.3016021251678467, 2.3149893283843994, 2.3272573947906494, 2.337588310241699, 2.352689743041992, 2.3650550842285156, 2.3788750171661377, 2.391205310821533, 2.4040026664733887, 2.418050765991211, 2.4329681396484375, 2.44647216796875, 2.4607110023498535, 2.477755069732666, 2.4911932945251465, 2.5057027339935303, 2.5252015590667725, 2.540255546569824, 2.539790391921997]





print(len(ranges))

x =[]
y =[]
t = math.pi * 3/ 2
for i in range(len(ranges)):
    if ranges[i] > 10.0:
        continue
    x.append(ranges[i]*math.cos(t*i/720))
    y.append(ranges[i]*math.sin(t*i/720))
plt.scatter(x,y)
plt.show()