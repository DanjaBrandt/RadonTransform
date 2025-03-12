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
        self.image_path = kwargs.get("image_path", "./data")

    def as_dict(self):
        """Returns the configuration parameters as a dictionary."""
        return self.__dict__

    def __repr__(self):
        """Returns a formatted string representation of the configuration."""
        return f"\n{pprint.pformat(self.__dict__)}"

