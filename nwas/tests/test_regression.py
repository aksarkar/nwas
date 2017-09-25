from nwas.models import *

def test_mediator_model():
    m = MediatorModel(numpy.zeros((10, 100)), numpy.zeros((10, 1)))

def test_mediator_model_fit():
    m = MediatorModel(numpy.zeros((10, 100)), numpy.zeros((10, 1)))
    m.fit()

def test_mediator_model_uninitialized_predict():
    with pytest.raises(ValueError):
        m = MediatorModel(numpy.zeros((10, 100)), numpy.zeros((10, 1)))
        m.predict(numpy.zeros((20, 100)))

def test_mediator_model_predict():
    m = MediatorModel(numpy.zeros((10, 100)), numpy.zeros((10, 1))).fit()
    mean, var = m.predict(numpy.zeros((20, 100)))
    assert mean.shape == [20, 1]
    assert var.shape == [20, 1]

def test_mediator_model_multi_response_predict():
    m = MediatorModel(numpy.zeros((10, 100)), numpy.zeros((10, 5))).fit()
    mean, var = m.predict(numpy.zeros((20, 100)))
    assert mean.shape == [20, 5]
    assert var.shape == [20, 5]

def test_mediator_model_two_instances():
    m0 = MediatorModel(numpy.zeros((10, 100)), numpy.zeros((10, 1)))
    m0.fit()
    m1 = MediatorModel(numpy.zeros((50, 500)), numpy.zeros((50, 1)))
    m1.fit()

def test_phenotype_model():
    m = PhenotypeModel(numpy.zeros((10, 100)), numpy.zeros((10, 1)),
                       numpy.zeros((10, 1)), numpy.zeros((10, 1)))

def test_phenotype_model_fit()
    m = PhenotypeModel(numpy.zeros((10, 100)), numpy.zeros((10, 1)),
                       numpy.zeros((10, 1)), numpy.zeros((10, 1)))
    m.fit()
