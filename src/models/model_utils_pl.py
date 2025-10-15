from __future__ import absolute_import

from src.models.transformer_pl import Transformer_pl
from src.models.transformer_pl_multiquery import Transformer_pl_MultiQuery

def create_model(conf, args):
    model_type = conf.model.model_type
    print("Creating model: ", model_type)
    use_multiquery = getattr(args, 'num_queries', 1) > 1
    if model_type == "transformer":
        model = Transformer_pl_MultiQuery(conf, args) if use_multiquery else Transformer_pl(conf, args)
    elif model_type == "transformer_vit16":
        model = Transformer_pl_MultiQuery(conf, args) if use_multiquery else Transformer_pl(conf, args)
    elif model_type == "transformer_dinov2_vitb14":
        model = Transformer_pl_MultiQuery(conf, args) if use_multiquery else Transformer_pl(conf, args)
    return model
