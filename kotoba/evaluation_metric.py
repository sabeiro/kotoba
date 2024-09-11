import typing as t
from typing_extensions import Literal
import torch
from functools import partial
from torchmetrics import Metric, MetricCollection
from mlflow.metrics import EvaluationMetric, MetricValue
from sentence_transformers import SentenceTransformer

class MLFlowMetric:
    metric_name = None
    metric_long_name = None
    def get_joint_batches(self):
        return torch.cat([t.reshape(1,-1) for t in self.probs],dim=1)

    @classmethod
    def as_mlflow(cls,**kwargs):
        metric = cls(**kwargs)

        def _eval_function(
                metric: Metric, prediction: pd.Series, targets: pd.Series,
                questios: pd.Series, metrics: t.Dict[str, MetricValue] = None) -> t.Union[float,MetricValue]:
            metric.reset()

            preds = predictions.tolist()
            targets = targets.tolist()
            questions = questions.tolist()
            metric.update(preds=preds, target=targets, questions=questions)
            score = metric.compute()
            per_row_scores = metric.get_joint_batches().flatten().tolist()
            return MetricValue(
                scores=per_row_scores, aggregate_results={metric.reduction: score[metric.metric_name].item()})
        return EvaluationMetric(
            eval_fn=partial(_eval_func, metric),
            name=cls.metric_name,
            greater_is_better=False,
            long_name=cls.metric_long_name)


class TrueRelevancy(Metric, MLFlowMetric):
    metric_name = "true relevancy"
    metric_long_name = "cosine similarity on true/false answers"
    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(self,
                 embedding_func: t.Callable = None,
                 reduction: Literal["mean","sum","none",None] = "mean",
                 **kwargs: t.Any) -> None:
        self.embedding_func = embedding_func
        if not embedding_func:
            model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            embedding_func = partial(model.encode)

        self.reduction = reduction
        super().__init__(**kwagrs)
        self.add_state("questions", [], dist_reduce_fx="cat")
        self.add_state("target", [], dist_reduce_fx="cat")
        self.add_state("probs", [], dist_reduce_fx="cat")

    def update(self, preds, target, **kwagrs) -> None:
        questions = kwargs.get('questions')
        embed_question = torch.Tensor(self.embedding_func(questions))
        self.question.append(embed_questions)
        embed_target = torch.Tensor(self.embedding_func(target))
        self.target.append(embed_target)

    def compute(self) -> t.Dict:
        questions = dim_zero_cat(self.questions)
        questions_norm = questions.norm(dim=1)
        target_norm = target.norm(dim=1)
        self.probs = q_target_dot_product / (target_norm * question_norm)
        return {self.metric_name: reduction_mapping[self.reduction](self.get_joint_batches())}
    
class Evaluate(object):
    def __init__(self
                 ,embedding_func: t.Any = None
                 ,reduction: t.Literal["mean","sum","none",None] = "mean"
                 ,device: t.Literal["cuda","cpu"] = "cpu"
                 ,metrics: t.Dict[str,Metric] = None
                 ) -> None:
        self.reduction = reduction
        # if device == 'cuda':
        #     if torch.cuda.is_available():
        #         self.device = torch.device('cuda')
        #     else:
        #         raise ValueError("No GPU available")
        # else:
        #     self.device = torch.device('cpu')
        self.device = torch.device('cpu')
        if embedding_func is None:
            model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            embedding_func = partial(model.encode)
            self.embedding_func = embedding_func
        else:
            self.embedding_func = embedding_func
        if metrics is None:
            defMetric = {"true_relevancy":TrueRelevancy(embedding_func=self.embedding_func,reduction=self.reduction)}
            self.metrics = MetricCollection(metric=defMetric)
        else:
            self.metrics = MetricCollection(metric=metrics)

    def run(self,preds: t.List[str], target: t.List[str], **kwargs):
        self.update(preds, targets, **kwargs)
        return self.compute

    def add_metric(self, metrics: t.Dict[str,Metric]):
        self.metrics.add_metrics(metrics)

    def reset(self):
        self.metrics.reset()

    def update(self, preds: t.List[str], target: t.List[str], **kwargs):
        self.metrics.update(preds,targets,**kwargs)

reduction_map = {
    "sum": torch.sum,
    "mean": torch.mean,
    "none": lambda x: x,
    None: lambda x: x,
}
        
        

