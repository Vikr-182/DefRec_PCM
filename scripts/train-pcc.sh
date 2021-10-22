#!/bin/zsh
#SBATCH --job-name=train-pcc
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=35
#SBATCH --output pcc.log
#SBATCH -G 4 -c 35
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

#module load cuda/11.0
#module load cudnn/7-cuda-11.0

DATASET_FILE1="/share1/vikrant.dewangan/dataset/pcc/pointda/"
DATASET_DST="/scratch/shapenets"

function setup {
    mkdir -p $DATASET_DST
    echo "Inside setup"

    scp -r ada:$DATASET_FILE1 $DATASET_DST
    cd /scratch/shapenets/pointda
    unzip PointDA_data.zip
}

echo "Copying"
#[ -d $DATASET_DST  ] || setup
setup
cd 

echo "Done copying"
#conda activate forecasting
echo "Done conda"
echo "LALA" >> lala.txt
cd /scratch/shapenets/pointda
#unzip PointDA_data.zip
cd ~/pcc/DefRec_and_PCM

#parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')
#parser.add_argument('--apply_PCM', type=str2bool, default=True, help='Using mixup in source')
#parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of train batch per domain')
#parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
#parser.add_argument('--supervised', type=str2bool, default=True, help='run supervised')
#parser.add_argument('--softmax', type=str2bool, default=False, help='use softmax')
#parser.add_argument('--use_DeepJDOT', type=str2bool, default=True, help='Use DeepJDOT')
#parser.add_argument('--DeepJDOT_head', type=str2bool, default=False, help='Another head for DeepJDOT')
#parser.add_argument('--DefRec_on_trgt', type=str2bool, default=True, help='Using DefRec in source')
#parser.add_argument('--DeepJDOT_classifier', type=str2bool, default=False, help='Using JDOT head for classification')

python3 PointDA/trainer.py --batch_size=64 --dataroot=/scratch/shapenets/pointda --epochs=75 --optimizer=ADAM --softmax=False --gpus=0,1,2,3 --src_dataset=shapenet --trgt_dataset=modelnet 
echo "Done Training"
