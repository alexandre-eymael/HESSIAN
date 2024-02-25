cd ..
wandb docker-run -v ./data:/app/data -v ./weights:/app/weights --gpus all \
hessian_docker_train \
--model_size small \
--save_freq 1 \
--epochs 1000 \
--lr 3e-4 \
--train_prop 0.8 \
--device cuda \
--img_size 224 \
--batch_size 64 \
--wandb_mode online \
--optimizer AdamW \
--seed 42