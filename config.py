import pprint

class Config:
    """
    Configuration class for Radon Transform analysis.

    This class stores parameters required for the analysis,
    allowing flexible initialization with default values.
    """

    def __init__(self, **kwargs):
        self.patch_size = kwargs.get("patch_size", 50)
        self.patch_step = kwargs.get("patch_step", 25)
        self.sigma = kwargs.get("sigma", 1)
        self.sigma_1d = kwargs.get("sigma_1d", 1)
        self.background_pixel_cutoff = kwargs.get("background_pixel_cutoff", 80)
        self.peak_threshold = kwargs.get("peak_threshold", 0.95)
        self.peak_prominence = kwargs.get("peak_prominence", 0.10)
        self.distance_threshold = kwargs.get("distance_threshold", 10)
        self.angle_threshold = kwargs.get("angle_threshold", 10)

    def as_dict(self):
        """Returns the configuration parameters as a dictionary."""
        return self.__dict__

    def __repr__(self):
        """Returns a formatted string representation of the configuration."""
        return f"\n{pprint.pformat(self.__dict__)}"

