import hydra
from omegaconf import DictConfig

try:
    import enefit

    env = enefit.make_env()
    iter_test = env.iter_test()

except ModuleNotFoundError:
    print("ModuleNotFoundError: enefit not found. Skipping test_api.py")


@hydra.main(config_path="../config/", config_name="predict")
def _main(cfg: DictConfig):
    counter = 0
    for (
        test,
        revealed_targets,
        client,
        historical_weather,
        forecast_weather,
        electricity_prices,
        gas_prices,
        sample_prediction,
    ) in iter_test:
        if counter == 0:
            print(test.head(3))
            print(revealed_targets.head(3))
            print(client.head(3))
            print(historical_weather.head(3))
            print(forecast_weather.head(3))
            print(electricity_prices.head(3))
            print(gas_prices.head(3))
            print(sample_prediction.head(3))
        sample_prediction["target"] = 0
        env.predict(sample_prediction)
        counter += 1

    print(f"Counter: {counter}")


if __name__ == "__main__":
    _main()
