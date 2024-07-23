import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import json

def get_emb():  
  emb = np.array([[ 4.45015449e-03, -2.89905127e-02, -3.98891643e-02,
          -4.35884949e-03,  8.89121741e-03,  1.96220987e-02,
          4.41063102e-03,  2.97842622e-02,  4.24957611e-02,
          -1.13303950e-02, -1.79393888e-02, -1.85662620e-02,
          6.36425149e-03, -5.07136472e-02, -2.10223682e-02,
          1.07639534e-02,  2.12146956e-02,  5.74890524e-03,
          1.61176734e-03, -1.74566638e-02,  6.70869043e-03,
          -3.28730270e-02, -6.87232381e-03, -1.04210246e-02,
          -6.91693556e-03, -7.13964831e-03,  3.16762701e-02,
          -9.75076016e-03, -2.40006931e-02,  2.60080192e-02,
          -2.99809370e-02, -1.19761257e-02, -1.45160686e-02,
          3.40624526e-02,  1.61193665e-02, -1.54642854e-02,
          1.11397421e-02, -1.97342597e-02, -7.10069900e-03,
          9.45886597e-03,  2.78869015e-03, -2.00871076e-03,
          5.82913868e-03,  1.79395231e-03, -9.08369012e-03,
          -9.25177988e-03, -1.86465755e-02, -9.74449888e-03,
          1.10192019e-02, -6.72306027e-03, -3.60774212e-02,
          3.12103424e-02,  2.26913821e-02,  5.07142395e-05,
          2.34816410e-02,  2.79993899e-02,  9.43991356e-03,
          6.19647675e-04,  2.86293719e-02,  8.04143958e-03,
          3.66192348e-02, -2.11802460e-02, -1.48543026e-02,
          1.65595207e-02,  1.77233610e-02,  8.23934563e-03,
          3.64607945e-03, -1.32743837e-02, -1.09443190e-02,
          -1.21379155e-03,  7.48157594e-03, -1.27214771e-02,
          1.73726492e-02, -4.63734344e-02, -3.03604063e-02,
          4.86031501e-03, -5.79397986e-03, -4.30004001e-02,
          -1.54001731e-02,  2.89608799e-02, -6.32700929e-03,
          -2.10513435e-02, -2.89985482e-02, -2.21955515e-02,
          1.10012908e-02, -2.99900472e-02,  3.05967033e-03,
          -9.41413455e-03, -1.03960456e-02,  4.66848072e-03,
          6.83014980e-03,  3.06159463e-02, -5.42072430e-02,
          5.13749421e-02, -4.48436514e-02,  2.44708266e-02,
          -5.95650263e-02,  1.14515886e-01,  2.44243369e-02,
          -1.54039124e-03, -5.64612448e-03, -2.14075278e-02,
          2.05561165e-02,  8.29332508e-03,  3.35779553e-03,
          2.00653411e-02, -2.43469331e-04, -6.22300878e-02,
          -1.68932024e-02, -6.58464711e-03, -1.46374032e-02,
          3.19682434e-02, -4.82072793e-02, -1.76243819e-02,
          -2.06091013e-02,  1.99569575e-02,  3.60438554e-03,
          2.16207765e-02, -2.84156688e-02,  9.58756730e-03,
          2.65453081e-03, -5.12059145e-02,  1.32895978e-02,
          -8.22904345e-04,  2.46804543e-02, -2.65062544e-02,
          7.44086970e-03, -8.19464959e-03, -1.85599420e-02,
          1.23193981e-02, -8.85888748e-03, -2.22809855e-02,
          -2.57738233e-02, -3.60139683e-02, -1.86854252e-03,
          1.45158572e-02,  1.86385028e-02, -2.67515481e-02,
          -2.05743546e-03,  2.65096333e-02,  2.54854672e-02,
          1.52405128e-02, -4.71924897e-03,  1.79199055e-02,
          4.95307371e-02,  2.85465010e-02,  1.14826122e-02,
          -1.74901634e-02,  1.19841099e-03,  2.66071106e-03,
          4.41418123e-03, -2.78894696e-02, -5.62728159e-02,
          -5.15504647e-03,  7.69220805e-03, -1.62498094e-02,
          -3.56166735e-02, -2.29265410e-02,  2.16955878e-03,
          -9.72433668e-03,  1.42064393e-02, -4.10045758e-02,
          5.73762832e-03,  2.13217046e-02, -3.84453381e-03,
          -3.80610824e-02, -1.37307658e-03, -1.51347173e-02,
          -2.05564732e-03, -2.43104305e-02, -3.20606828e-02,
          4.02994379e-02,  2.04340462e-02, -2.32184809e-02,
          7.91019935e-04, -7.82529823e-03, -3.31472866e-02,
          1.19122779e-02,  3.41824219e-02,  4.37753499e-02,
          3.69915590e-02,  1.47804739e-02, -1.38057601e-02,
          -1.77388284e-02,  3.91002968e-02,  3.27121140e-03,
          -3.88966575e-02, -1.24924304e-02,  5.67019591e-03,
          -3.18806805e-02, -6.61136210e-03,  5.24239019e-02,
          6.55688113e-03,  1.14142578e-02, -2.59878598e-02,
          1.51388524e-02, -2.54164133e-02,  2.43880711e-02,
          3.72751951e-02, -5.02612777e-02, -3.18231732e-02,
          7.11319735e-03,  9.93019901e-03, -4.37125899e-02,
          1.08276214e-02,  2.41645426e-02, -2.53605060e-02,
          3.80046070e-02, -1.42527912e-02,  4.90478892e-03,
          -8.67526513e-03,  3.22401035e-03,  1.69170275e-02,
          -3.98941291e-03,  2.89673563e-02, -2.12466419e-02,
          2.90612830e-03,  5.32763191e-02, -3.12720314e-02,
          1.52835967e-02, -1.61546897e-02,  5.10033080e-03,
          -2.32564975e-02, -1.43825561e-02, -9.78703517e-03,
          2.55952775e-02,  9.21370368e-03,  3.17141972e-02,
          1.65435337e-02,  9.95563250e-03, -8.47112201e-03,
          8.12917389e-03, -1.96684469e-02, -7.41976034e-03,
          4.55199648e-03,  9.81630664e-03,  2.36819424e-02,
          2.94342812e-04,  1.35593219e-02,  3.25389439e-03,
          -1.44168790e-02,  1.89945176e-02,  1.91436745e-02,
          1.83477886e-02, -7.73326028e-03, -4.20891009e-02,
          2.58679800e-02,  2.69985851e-02, -4.78608708e-04,
          2.48625912e-02,  1.02675948e-02, -3.46979573e-02,
          7.80541450e-03,  1.68781425e-03,  2.51874737e-02,
          1.55765917e-02,  2.28464026e-02,  1.83212012e-02,
          6.03634492e-02,  6.50552958e-02, -3.97647396e-02,
          7.65871396e-03,  3.69136594e-02,  1.32306796e-02,
          2.85058804e-02, -8.72808974e-03, -3.36924195e-02,
          -1.09482873e-02, -3.97077063e-03,  4.16664127e-03,
          -1.01734521e-02, -1.32535845e-02,  1.04415673e-03,
          1.13917375e-02,  2.13872194e-02,  9.37185250e-03,
          2.35289875e-02, -7.66381901e-03,  1.53728575e-02,
          -9.09238588e-03,  1.13019776e-02,  3.13561899e-03,
          -2.37532873e-02,  1.63044892e-02, -1.20576154e-02,
          2.40252931e-02, -6.34477884e-02,  6.98132291e-02,
          5.22622094e-02, -3.53862271e-02,  1.10164436e-03,
          1.48783224e-02, -4.94018849e-03,  4.00454402e-02,
          -4.52604368e-02, -4.37972061e-02, -1.54707581e-02,
          1.49015263e-02,  1.64569006e-03, -3.97248827e-02,
          9.05441400e-03, -9.13873035e-03, -2.06992384e-02,
          -2.84186751e-02,  6.87452545e-03,  9.28680785e-03,
          9.52081569e-03,  1.42135620e-02,  2.62764692e-02,
          5.24042547e-02,  3.22060287e-02,  1.30960522e-02,
          2.83752079e-03,  3.80964726e-02, -4.60869148e-02,
          2.21013185e-02,  2.65777111e-02, -1.27289994e-02,
          1.81649216e-02, -6.01742044e-03,  8.26509297e-03,
          -4.16465802e-03,  4.90287952e-02, -3.37466598e-02,
          -1.02876434e-02,  8.31484050e-03,  2.10221708e-02,
          1.40363276e-02, -1.40066938e-02,  4.40760627e-02,
          -1.22468069e-01,  5.88166118e-02, -8.48994963e-03,
          4.22871634e-02, -4.23615426e-03, -3.59589867e-02,
          2.99550630e-02, -3.02315764e-02,  9.91008896e-03,
          -1.18597221e-04, -1.01033114e-02, -1.63211487e-02,
          -4.59054895e-02,  2.24217437e-02, -2.70271357e-02,
          -3.99246905e-03,  4.47370186e-02, -2.83185728e-02,
          1.38546675e-02,  1.39064044e-02,  2.96354806e-03,
          -1.99614875e-02,  1.51638389e-02, -1.24210902e-02,
          1.50780678e-02,  1.86800361e-02, -2.86061503e-02,
          -1.40166339e-02,  6.29950315e-03,  1.39358137e-02,
          -2.59884950e-02,  3.14422511e-02, -4.85726818e-03,
          1.60819981e-02,  7.86760822e-04, -4.11619097e-02,
          -5.30318182e-04,  2.32106317e-02, -2.66444627e-02,
          -1.21119628e-02,  3.34070586e-02,  2.69958265e-02,
          4.63798642e-02,  1.05416998e-02, -2.84894966e-02,
          5.92245441e-03, -8.47173668e-03,  2.21162029e-02,
          1.14669781e-02, -1.26343472e-02,  2.12939596e-03,
          -9.61762201e-03,  2.52003558e-02,  1.59576051e-02,
          3.75903174e-02, -3.09117474e-02,  1.73320323e-02,
          -2.57445779e-02,  2.09594965e-02,  2.76705530e-03,
          1.28984721e-02, -9.91428364e-03,  1.94068737e-02,
          -6.30396605e-03,  7.87652843e-03,  3.98091273e-03,
          1.29270153e-02,  4.26652804e-02, -2.87519768e-03,
          8.82776175e-03, -7.15002511e-03,  2.05005966e-02,
          1.06347017e-02, -1.48282899e-02,  2.32419427e-02,
          -2.95598060e-03, -9.33321286e-03, -5.87110333e-02,
          -1.90253537e-02, -2.21044719e-02, -8.12769867e-03,
          5.36459172e-03,  2.17061080e-02, -3.20130959e-02,
          -1.34330837e-03,  1.40477512e-02,  2.02447665e-03,
          1.90210901e-02,  2.27239542e-03,  2.58041862e-02,
          3.70804369e-02, -3.06352023e-02, -4.87531070e-03,
          -2.09392440e-02,  1.74303669e-02, -7.25751463e-03,
          2.73828171e-02,  5.68727776e-03,  1.21287014e-02,
          -3.25142778e-02, -5.11461869e-02, -1.76004262e-03,
          -9.90900304e-03, -2.90980004e-02, -3.64695117e-02,
          -9.68320202e-03, -1.41199706e-02,  4.44145091e-02,
          4.70991544e-02, -9.39113903e-04,  3.05248164e-02,
          2.12705694e-03,  9.20164306e-03, -3.13792042e-02,
          -5.20001119e-03,  2.80829910e-02, -9.83197894e-03,
          3.49273486e-03,  3.46274301e-02,  2.26577222e-02,
          -6.26148731e-02,  3.37562640e-03,  1.92963774e-03,
          -1.04359247e-01,  1.91719527e-03, -1.45439361e-03,
          8.99751671e-03, -2.27540219e-03,  2.14986913e-02,
          2.83112447e-03, -3.85434669e-03, -2.06403881e-02,
          -3.15531194e-02,  1.36746410e-02,  2.64487416e-02,
          -6.86730593e-02,  1.38197336e-02, -1.78956669e-02,
          4.36957292e-02,  2.62097511e-02, -1.32859619e-02,
          -3.40083730e-03,  7.65039772e-03,  5.50250243e-03,
          4.72705672e-03, -2.65239961e-02, -3.07938270e-02,
          8.05258787e-06,  4.22949949e-03,  3.02906474e-03,
          1.89244114e-02, -1.29956976e-02,  4.33549564e-03,
          -3.55428760e-03,  4.24335431e-03, -1.55355427e-02,
          -2.18153605e-03,  4.17716280e-02,  3.75917973e-03,
          -1.85229871e-02, -4.05204482e-02, -1.75183453e-02,
          1.50714414e-02,  1.22038322e-02,  2.36048431e-33,
          -2.60259258e-03, -9.36569832e-03,  5.84070617e-03,
          3.58923897e-02,  1.08829234e-02,  1.70708112e-02,
          4.79532406e-03, -1.78119242e-02, -1.04873544e-02,
          1.04686175e-03, -3.83881852e-02, -1.01732071e-02,
          6.56425196e-04, -2.92319711e-02,  1.45151494e-02,
          3.69927660e-02, -8.78997985e-03, -1.70536060e-03,
          -2.72527849e-03,  6.16632542e-03,  2.10952275e-02,
          3.76902195e-03, -3.79007198e-02,  6.57932321e-03,
          -1.69635471e-02,  1.07767638e-02, -4.11467766e-03,
          -8.31058063e-03,  1.73516981e-02,  1.83953773e-02,
          1.00284368e-02, -1.02935564e-02,  4.48381603e-02,
          1.82367750e-02,  1.86645724e-02, -1.85761917e-02,
          4.31111502e-03, -4.42832708e-04, -2.46741846e-02,
          4.44256514e-03, -2.94747390e-03, -1.03870304e-02,
          3.21468934e-02,  2.63079591e-02, -1.28408801e-02,
          -3.81002156e-03, -5.54827624e-04, -4.40934785e-02,
          1.11768413e-02, -8.26051179e-03, -1.51608884e-03,
          2.13574599e-02,  3.27811800e-02, -4.83348630e-02,
          -2.01373291e-03,  1.04349349e-02, -6.61588684e-02,
          -7.38775879e-02,  2.23315414e-02,  1.39844809e-02,
          1.81404501e-02,  4.22135927e-02,  2.22733859e-02,
          -8.37282278e-03, -1.58792045e-02,  1.59301031e-02,
          -1.03376284e-02,  2.20559631e-02,  8.60543083e-03,
          -1.19523685e-02, -5.80205321e-02, -1.44396047e-03,
          -1.27335312e-02,  1.81837771e-02,  2.81889997e-02,
          -1.02947059e-03, -1.22019583e-02, -9.15660698e-04,
          1.59564726e-02, -1.26515804e-02, -6.80480152e-05,
          1.21276770e-02,  3.21222022e-02,  9.57298651e-03,
          2.34918427e-02,  4.74055633e-02, -1.16552692e-02,
          1.53134260e-02,  1.30853970e-02, -1.30519718e-02,
          -2.18303762e-02,  2.88903108e-03,  6.41203206e-03,
          -3.19487503e-04,  1.91162918e-02,  6.53161556e-02,
          8.56500398e-03,  2.13221367e-02, -4.03581467e-03,
          -1.71075426e-02, -2.31356565e-02,  8.94622318e-03,
          1.33259781e-03, -2.94181444e-02, -2.79667787e-03,
          -1.26581583e-02, -3.41298501e-03, -3.20861489e-03,
          -3.03021930e-02, -2.03261934e-02, -1.25518190e-02,
          7.51832426e-02,  3.44381928e-02, -4.87782946e-03,
          -8.35128408e-03,  2.68825591e-02, -2.24163644e-02,
          -2.05869190e-02, -1.65340193e-02,  1.36266919e-02,
          -5.14297164e-04, -6.30668271e-03,  6.22461224e-03,
          -1.98250916e-02,  2.68193595e-02,  1.33279292e-02,
          1.09922271e-02, -2.71267770e-03,  1.45420507e-02,
          -2.15888838e-03,  2.43351944e-02, -6.99330727e-03,
          -1.23847155e-02,  1.72296520e-02,  1.25938235e-02,
          1.05150361e-02,  2.06294693e-02, -1.37038603e-02,
          -1.17308097e-02, -3.55217755e-02,  9.52999294e-03,
          3.66363749e-02, -2.72038946e-04, -1.20195728e-02,
          1.13929529e-02, -3.03304289e-02,  4.94673196e-03,
          -2.76524276e-02,  3.05948798e-02, -2.89632790e-02,
          2.28249542e-02,  1.11988164e-03,  2.69709714e-02,
          -5.49166882e-03, -1.88776329e-02, -1.83197595e-02,
          1.56356785e-02,  1.52846668e-02, -2.24260874e-02,
          1.62876006e-02, -3.63245863e-03, -2.18141675e-02,
          -1.97326820e-02,  1.08254682e-02,  2.26229808e-04,
          -3.14583443e-02, -1.82108805e-02,  1.83530897e-02,
          -1.91261282e-03,  1.27931414e-02, -2.14035367e-03,
          3.44225904e-03,  4.72800341e-03,  4.42005834e-03,
          1.50743453e-02, -2.69684233e-02,  6.82466337e-03,
          -7.00389147e-02,  1.29581867e-02, -4.63581271e-02,
          -1.74308456e-02, -3.80314561e-03,  7.82460254e-03,
          -7.49335438e-03,  6.11753650e-02,  4.75559495e-02,
          -2.06569787e-02,  3.55592892e-02, -4.35754284e-02,
          1.09387990e-02, -3.85149755e-02,  2.51014046e-02,
          2.33092718e-02, -2.35601068e-02, -3.84794170e-04,
          -1.76973117e-03,  1.43241752e-02, -3.28296237e-02,
          -1.23837357e-02, -1.65266003e-02,  1.19157853e-02,
          -4.91251983e-02, -1.04352292e-02, -8.65002349e-03,
          9.95829143e-03,  1.62468217e-02, -5.30827157e-02,
          6.97616627e-03, -1.20238420e-02,  3.13457660e-02,
          -1.42401960e-02,  3.08980104e-02, -3.37388702e-02,
          -2.23557632e-02, -4.21541091e-03, -3.69132273e-02,
          2.13045888e-02,  2.82772221e-02,  6.31117746e-02,
          1.18411705e-02,  1.07754879e-02,  1.01349419e-02,
          2.27780025e-02, -7.22516049e-03, -1.50985103e-02,
          -1.63407568e-02,  2.32298896e-02, -2.89782472e-02,
          7.45718321e-03, -7.41018821e-03, -1.82995982e-02,
          -1.73546579e-02,  1.26802232e-02, -6.66742492e-03,
          -4.45020851e-03,  1.56143252e-02, -5.43812029e-02,
          -2.55678850e-03,  2.31747292e-02,  6.01570494e-03,
          2.66911201e-02, -2.50963308e-02,  7.39068305e-03,
          -8.62636697e-03, -2.48387866e-02,  8.87084007e-03,
          2.73453817e-03, -1.15386322e-02,  2.65196972e-02,
          -2.64699645e-02, -3.91907617e-02, -7.69379735e-03,
          3.64944665e-03, -1.66307930e-02, -1.19670648e-02,
          -8.24410282e-03, -3.25694121e-03,  2.51292381e-02,
          2.22818553e-02,  1.07823918e-02, -3.83924916e-02,
          -2.06056181e-02, -1.56761892e-03,  3.23194638e-03,
          2.23886594e-02, -1.32844746e-02, -3.59200388e-02,
          2.61462238e-02, -1.70545224e-02,  3.33530875e-03,
          2.37896703e-02, -2.14133896e-02,  2.58927401e-02]], dtype=np.float32)
  return emb 

def normalize_timeline(numbers):
  # Normalize to range 0-1
  min_val = min(numbers)
  max_val = max(numbers)
  normalized_numbers = (np.array(numbers) - min_val) / (max_val - min_val)
  # Scale to range 0-4
  scaled_numbers = normalized_numbers * 3
  # Round to nearest integer
  regularized_numbers = np.round(scaled_numbers).astype(int)
  return regularized_numbers.tolist()

def frust_index(d, emb):
  em = []
  model_sent_trans = SentenceTransformer('all-distilroberta-v1')
  for i in range(len(d)):
    e = model_sent_trans.encode(d['Utterance'].iloc[i])
    c = cosine(emb[0], e)
    em.append(c)
  d['cos']  = em

  stu_frust = list(d[(d['cos']<0.6) & (d['Speaker'] == 'student')].index)
  return stu_frust


def get_not_understanding_idx(data):
  q_idx = list(data[(data['Speaker'] == 'student') & (data['DA'] == 'Questions') & (data['Utterance'].str.contains('\?')) & (~(data['Utterance'].str.contains('right?')))].index)
  n_idx = frust_index(data, get_emb())
  idx = q_idx + n_idx
  return list(set(idx))

def extract_and_evaluate_problems_with_ending(data):
    problem_indexes =  get_not_understanding_idx(data)
    if len(problem_indexes) == 0:
        return pd.DataFrame()
    problems = []

    for index in problem_indexes:
        if index + 1 < len(data) and data.loc[index + 1, 'DA'] == 'Explanation':
            explanation_end_index = index + 1
            while explanation_end_index + 1 < len(data) and data.loc[explanation_end_index + 1, 'DA'] == 'Explanation':
                explanation_end_index += 1

            next_utterance_index = explanation_end_index + 1
            next_utterance = data.loc[next_utterance_index, 'Utterance'] if next_utterance_index < len(data) else None
            next_da_final = data.loc[next_utterance_index, 'DA'] if next_utterance_index < len(data) else None

            # Additional logic to manage the continued explanations
            if next_utterance_index not in problem_indexes and next_utterance_index + 1 < len(data):
                if data.loc[next_utterance_index + 1, 'DA'] == 'Explanation':
                    explanation_end_index = next_utterance_index + 1
                    while explanation_end_index + 1 < len(data) and data.loc[explanation_end_index + 1, 'DA'] == 'Explanation':
                        explanation_end_index += 1
                    next_utterance_index = explanation_end_index + 1
                    next_utterance = data.loc[next_utterance_index, 'Utterance'] if next_utterance_index < len(data) else None
                    next_da_final = data.loc[next_utterance_index, 'DA'] if next_utterance_index < len(data) else None

            # Get 10 lines of context before the problem utterance
            context_start_index = max(0, index - 10)
            context = " ".join(data.loc[context_start_index:index - 1, 'Utterance'])

            problems.append({
                "Problem": f"problem {len(problems) + 1}",
                "DA_Type": data.loc[index, 'DA'],  # Store the DA type of the problem
                "time": data.loc[index, 'Utterance start time (milliseconds)'],
                "Not Understanding": data.loc[index, 'Utterance'],
                "Explanation": " ".join(data.loc[index + 1:explanation_end_index + 1, 'Utterance']),
                "Next Utterance": next_utterance,
                "Next DA_final": next_da_final,
                "Context": context
            })

    return pd.DataFrame(problems)

def analyze_with_gpt(client, problem, explanation, context):


    system_prompt_1 =  f"""You are an expert in learning and you have multiple years in tutoring students on their homework.
    In this task, you will be given transcripts of a tutor and a student having a tutoring session. Your task is to analyze whether the tutor's answer to student's questions and confusions are
    *factually correct* and if the answers are *pedagogically sound, sufficient and direct responses to students' question*.
    Step by step instructions:
    1. First, read in the student's problem: {problem}.
    2. Then, read the lines before the student's problem and think step by step to generate a contextualized question: {context}.
    3. Then, read the tutor's explanation: {explanation}.
    4. Think step by step to answer if the tutor's explanation is factually correct and pedagogically sound, sufficient and direct responses given the contextualized question.
    5. Then think as the student who's having this question. Would you say that your answer is fully answered? """

    prompt = f"""{system_prompt_1}\nAnalyze the tutor's explanation and answer the following questions:\n1. Is the tutor's explanation factually correct?\n2.
    Is the tutor's explanation aligned and directly related to what the student is asking? For both questions, output 1 if your answer is yes, output 0 if your answer is no.
    Format your answer as JSON format, following the following xample json format: \"answer_1\": 1,  \"reason_1\": \"reason for question 1\", \"answer_2\": 0, \"reason_2\": \"reason for question 2\""""

    messages = [{"role": "system", "content": prompt} ]
    response = client.chat.completions.create(
        model = "gpt-4",
        messages = messages,
        temperature =0
    )
    r = response.choices[0].message.content
    return  r


def get_problem_analysis(client, data):
  problems_df = extract_and_evaluate_problems_with_ending(data)
  if len(problems_df) == 0:
    return pd.DataFrame(), -1, -1
  align = []
  factually = []
  for index, row in problems_df.iterrows():
    r = analyze_with_gpt(client, row['Not Understanding'], row['Explanation'], row['Context'])
    align.append(json.loads(r)['answer_2'])
    factually.append(json.loads(r)['answer_1'])
  problem  = [2 - int(x) - int(y) for x, y  in zip(factually, align)]
  time = problems_df['time']
  df = pd.DataFrame({'problem': problem, 'time': time}).sort_values(by='time', ascending=True)
  if len(align) == 0:
    align_p = -1
  else:
    align_p = sum([int(x) for x in align])/len(align)
  if len(factually) == 0:
    factually_p = -1
  else:
    factually_p = sum([int(x) for x in factually])/len(factually)
  return df, align_p, factually_p


def effective(data):
  duration = data['Utterance end time (milliseconds)'].iloc[-1]
  start = 0
  end = 1
  l = []
  idx = []
  c = 0
  q = 0
  for i in range(len(data)):
    if data['Utterance end time (milliseconds)'].iloc[i] > (end*data['Utterance end time (milliseconds)'].iloc[-1]/20):
        for j in range(start, i+1):
            if data['Speaker'].iloc[j] == 'tutor' and data['DA'].iloc[j].find('Question') != -1:
                q +=1
                if j+1 <len(data):
                  if data['Speaker'].iloc[j+1] == 'student':
                    c+=1
        if q == 0:
          l.append(1)
        else:
          l.append(1-c/q)
        start = i+1
        end = end+1

  for j in range(start, len(data)-1):
    if data['Speaker'].iloc[j] == 'tutor' and data['DA'].iloc[j].find('Question') != -1:
      q +=1
      if j+1 <len(data):
        if data['Speaker'].iloc[j+1] == 'student':
          c+=1
  if q == 0:
    l.append(1)
  else:
    l.append(1-c/q)

  return(l, np.mean(l))


def get_instruction(client, data):
  effective_list, effective_score = effective(data)
  df, align_p, factually_p =  get_problem_analysis(client, data)

  if len(df) == 0:
    slot_size = data['Utterance end time (milliseconds)'].iloc[-1] / 20

    # Create a new column for the slot number
    df['slot'] = (df['time'] / slot_size).astype(int)

    # Group by the slot and sum the values
    slot_sums = df.groupby('slot')['problem'].sum().reset_index()

    all_slots = pd.DataFrame({'slot': range(20)})
    slot_sums = all_slots.merge(slot_sums, on='slot', how='left').fillna(0)

    l = all_slots['problem']+effective_list
    tl = normalize_timeline(list(l))

    return tl, effective_score, align_p, factually_p

  slot_size = data['Utterance end time (milliseconds)'].iloc[-1] / 20

  # Create a new column for the slot number
  df['slot'] = (df['time'] / slot_size).astype(int)

  # Group by the slot and sum the values
  slot_sums = df.groupby('slot')['problem'].sum().reset_index()

  all_slots = pd.DataFrame({'slot': range(20)})
  slot_sums = all_slots.merge(slot_sums, on='slot', how='left').fillna(0)

  l = slot_sums['problem']+effective_list
  tl = normalize_timeline(list(l))
  return tl, effective_score, align_p, factually_p