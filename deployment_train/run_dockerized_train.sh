wandb docker-run \
-v data_volume:/app/data \
--gpus all \
hessian_docker_train \
--model_size small \
--save_freq 5 \
--epochs 1000 \
--lr 3e-4 \
--train_prop 0.8 \
--device cuda \
--img_size 224 \
--batch_size 32 \
--wandb_mode online \
--optimizer AdamW \
--seed 42 \
--load_all_in_ram False
