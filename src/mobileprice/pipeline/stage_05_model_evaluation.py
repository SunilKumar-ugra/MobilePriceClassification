from mobileprice import logger
from mobileprice.config import ConfigurationManager
from mobileprice.components.model_evaluation import ModelEvaluation
from dotenv import load_dotenv


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        load_dotenv()
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(eval_config)
        evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        STAGE_NAME = "Evaluation stage"
        logger.info(f"*******************")
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        model_evalution = EvaluationPipeline()
        model_evalution.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
