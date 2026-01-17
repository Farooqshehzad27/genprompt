python train_genprompt.py --domains biggan crn cyclegan deepfake gaugan glow imle san stargan_gf stylegan --clip_model ViT-L/14 --n_clusters 12 --batch_size 64 --epochs_per_domain 10 --use_cached --output_dir experiments/validation_run_10_domains

python evaluate.py --checkpoint experiments/validation_run_10_domains/checkpoints/checkpoint_stylegan.pt --clip_model ViT-L/14 --domains biggan crn cyclegan deepfake gaugan glow imle san stargan_gf stylegan --test_domains whichfaceisreal wild --output_dir evaluation_results/full_evaluation

python evaluate.py --checkpoint experiments/validation_run_10_domains/checkpoints/checkpoint_stylegan.pt --clip_model ViT-L/14 --domains whichfaceisreal --output_dir evaluation_results/generalization_only
