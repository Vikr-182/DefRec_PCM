#!/bin/zsh
#SBATCH --job-name=supf-src-trgt-train-pcc
#SBATCH --time=1-72:00:00
#SBATCH --mincpus=5
#SBATCH --output supf-src-trgt-pcc.log
#SBATCH -G 1 -c 10
#SBATCH --mail-type=ALL --mail-user=vikrant.dewangan@research.iiit.ac.in

#module load cuda/11.0
#module load cudnn/7-cuda-11.0

DATASET_FILE1="/share1/vikrant.dewangan/dataset/pcc/pointda/"
DATASET_DST="/scratch/shapenets"

function setup {
    mkdir -p $DATASET_DST
    echo "Inside setup"

    scp -r ada:$DATASET_FILE1 $DATASET_DST
}

echo "Copying"
#[ -d $DATASET_DST  ] || setup
setup
cd 

echo "Done copying"
echo "Done conda"
echo "LALA" >> lala.txt
cd /scratch/shapenets/pointda
unzip PointDA_data.zip
cd ~/pcc/DefRec_and_PCM
#parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')
#parser.add_argument('--DeepJDOT_head', type=str2bool, default=False, help='Another head for DeepJDOT')
#parser.add_argument('--DefRec_on_trgt', type=str2bool, default=True, help='Using DefRec in source')
#parser.add_argument('--DeepJDOT_classifier', type=str2bool, default=False, help='Using JDOT head for classification')
python3 PointDA/trainer.py --dataroot /scratch/shapenets/pointda --batch_size 16 --supervised False --softmax True --DefRec_on_src True --DefRec_on_trgt True
echo "Done Training"
