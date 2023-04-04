# Run the perfect model on different models
for model in "bert" "roberta_base" "t5_small"; do \

        python run_clm.py configs/perfect_model_${model}.json
        
    done

done