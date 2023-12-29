python test_time_pstarc.py --gpu_id 0 --opt_fe --K 5  --tta_bs 128 --tta_method pstarc --lamda 1.0 --dshift r2c --dataset domainnet126 --tta_lr 2e-4
python test_time_pstarc.py --gpu_id 0 --opt_fe --K 5  --tta_bs 128 --tta_method pstarc --lamda 1.0 --dshift r2p --dataset domainnet126 --tta_lr 2e-4
python test_time_pstarc.py --gpu_id 0 --opt_fe --K 5  --tta_bs 128 --tta_method pstarc --lamda 1.0 --dshift p2c --dataset domainnet126 --tta_lr 2e-4
python test_time_pstarc.py --gpu_id 0 --opt_fe --K 5  --tta_bs 128 --tta_method pstarc --lamda 1.0 --dshift c2s --dataset domainnet126 --tta_lr 2e-4
python test_time_pstarc.py --gpu_id 0 --opt_fe --K 5  --tta_bs 128 --tta_method pstarc --lamda 1.0 --dshift s2p --dataset domainnet126 --tta_lr 2e-4
python test_time_pstarc.py --gpu_id 0 --opt_fe --K 5  --tta_bs 128 --tta_method pstarc --lamda 1.0 --dshift r2s --dataset domainnet126 --tta_lr 2e-4
python test_time_pstarc.py --gpu_id 0 --opt_fe --K 5  --tta_bs 128 --tta_method pstarc --lamda 1.0 --dshift p2r --dataset domainnet126 --tta_lr 2e-4