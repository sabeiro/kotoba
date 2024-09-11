import shap
import xgboost
import transformers
import sklearn
import numpy as np
import json
import xgboost
import matplotlib
matplotlib.use('Qt5Agg')

X, y = shap.datasets.california()
model = xgboost.XGBRegressor().fit(X, y)
y_pred = model.predict(X)
explainer = shap.Explainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap.dependence_plot("Latitude", shap_values, X,interaction_index="Population")
shap.plots.force(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0, :], matplotlib = True)
shap.decision_plot(explainer.expected_value, shap_values[1], X.columns)

shap_values = explainer(X)
shap.plots.waterfall(shap_values[0])
shap.plots.scatter(shap_values[:, "Latitude"], color=shap_values)
shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)

# svm = sklearn.svm.SVC(kernel='rbf', probability=True)
# svm.fit(X, y)
# explainer = shap.KernelExplainer(svm.predict_proba, X, link="logit")
# shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")
# shap.TreeExplainer(model).shap_interaction_values(X)

model = transformers.pipeline('sentiment-analysis', return_all_scores=True)
explainer = shap.Explainer(model)
shap_values = explainer(["What a great movie! ...if you have no taste."])
shap.plots.text(shap_values[0, :, "POSITIVE"])

background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(x_test[1:5])
shap.image_plot(shap_values, -x_test[1:5])

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
model = VGG16(weights='imagenet', include_top=True)
X,y = shap.datasets.imagenet50()
to_explain = X[[39,41]]
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)

def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)
e = shap.GradientExplainer(
    (model.layers[7].input, model.layers[-1].output),
    map2layer(X, 7),
    local_smoothing=0 # std dev of smoothing noise
)
shap_values,indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
shap.image_plot(shap_values, to_explain, index_names)

