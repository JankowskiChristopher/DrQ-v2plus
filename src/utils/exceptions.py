class Exceptions:
    """
    Class responsible for handling exceptions.
    """

    @staticmethod
    def stable_baselines3_version_exception() -> None:
        """
        Raise an exception if the version of stable-baselines3 is not the right one.
        """
        import stable_baselines3 as sb3

        if sb3.__version__ < "2.0":
            raise ValueError(
                """Ongoing migration: run the following command to install the new dependencies:
                poetry run pip install "stable_baselines3==2.0.0a1"""
            )
