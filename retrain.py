"""Simple retraining entry point.

Same flow as train.py. The promotion logic compares validation MAE and MSE with the active model and
does not replace it unless both improve. To retrain on data landed by the Kafka stream:

    python retrain.py --raw data/stream/visits
"""

from train import main


if __name__ == "__main__":
    main()
