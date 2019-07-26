python train.py --lr=0.01 --load=models/state_dict/07261341/0epo_0010000step.ckpt

for((i=1;i<100;i++))
do
    time=`ls -l models/state_dict/ | tail -n 1 | awk '{print $9}'`
    filename=`ls -l models/state_dict/${time}/  | tail -n 1 | awk '{print $9}'`
    model_path=models/state_dict/${time}/${filename}
    
    echo -e '\ntest: '${model_path}
    python test.py --load=${model_path}
    echo -e '\ntrain: '${model_path}
    python train.py --load=${model_path} --lr=0.01
done
