import pandas as pd

import os,sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# from baseline_methods.shapley_value import collect_shap_deep_explanations
# from treatment_prediction.nlp.nlp_data_utils.simulation import run_simulation
from models.Qmod_baseline import *
from parse_args_nlp import parse_args
from nlp_data_utils.nlp_dataset import *
from utils_nlp import load_pretrained_backbone_models
from nlp_data_utils.nlp_dataset import split_train_valid_test_df

if __name__ == "__main__":
        args = parse_args()
        
        train_dataset, valid_dataset, test_dataset, feat_range_mappings, id_attr, outcome_attr, treatment_attr, text_attr, count_outcome_attr, lang = construct_dataset_main(args)

        # device = ("cuda" if torch.cuda.is_available() else "cpu")

        # if args.dataset_name == "music":
        #         raw_df = pd.read_csv(os.path.join(args.data_folder, 'music.csv'))
        #         df, offset =run_simulation(raw_df, propensities=[0.8, 0.6], 
        #                                         beta_t=1.0, 
        #                                         beta_c=50.0,
        #                                         gamma=1.0   ,
        #                                         cts=True)


        # train_df, valid_df, test_df = split_train_valid_test_df(df)
        
        mod = QNet_baseline(train_dataset, valid_dataset, test_dataset, args.method,  batch_size = 4, # batch size for training
                a_weight = 0.1,  # loss weight for A ~ text
                y_weight = 0.1,  # loss weight for Y ~ A + text
                mlm_weight=1.0,  # loss weight for DistlBert
                modeldir=args.log_folder) # directory for saving the best model

        # load_pretrained_backbone_models(mod, os.path.join(args.log_folder, "_bestmod.pt"))

        # for p in mod.model.distilbert.parameters():
        #         p.requires_grad = False
        
        # if not args.posthoc:
        mod.train(epochs=20,   # the maximum number of training epochs
                learning_rate = 5e-4)  # learning rate for the training

        
                # Q0, Q1, A, Y, _ = mod.get_Q(test_df['text'], test_df['T'], test_df['C'], test_df['Y'])
        # else:
        #         mod.model = torch.load(args.log_folder+'_bestmod.pt')
        #         collect_shap_deep_explanations(mod.model, (mod.train_text[0:100], mod.train_confounder[]), test_data)

