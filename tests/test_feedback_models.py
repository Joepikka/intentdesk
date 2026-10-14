import numpy as np

from intentdesk.feedback_models import classification, regression, estimators, predict_batches


def test_declared_estimators_cover_requested_supervised_algorithms():
    models = estimators()
    assert set(models)=={"logistic_regression","linear_svm","decision_tree","random_forest","knn"}
    assert models["knn"][1]=="dense" and models["linear_svm"][1]=="sparse"


def test_metrics_keep_rating_regression_separate_from_classification():
    result = classification([1,2,3,4,5],[1,2,3,4,1])
    assert result["accuracy"]==.8 and result["confusion_matrix"][4][0]==1
    values = regression([1,2,3,4,5],np.array([1,2,3,4,6]))
    assert values["mae_stars"]==.2 and values["prediction_clipping"].startswith("none")


def test_predict_batches_bounds_inference_memory():
    class Model:
        def predict(self,x):
            assert len(x)<=128
            return x[:,0]
    x = np.arange(300).reshape(300,1)
    assert np.array_equal(predict_batches(Model(),x),x[:,0])
