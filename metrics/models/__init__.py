from .gaussian import Gaussian
from .funnel import Funnel
from .squiggle import Squiggle

# from .rosenbrock import Rosenbrock
from .hybrid_rosenbrock import HybridRosenbrock as Rosenbrock
from .gaussian_mixtures import NineGaussians, GMM, TwoGaussians
from .general_mixture import GeneralMixture
from .logistic_regression import BayesianLogisticRegression
from .distribution import Distribution

__all__ = [
    "NineGaussians",
    "GMM",
    "Gaussian",
    "TwoGaussians",
    "Funnel",
    "Squiggle",
    "Rosenbrock",
    "BayesianLogisticRegression",
    "Distribution",
    "GeneralMixture",
]
