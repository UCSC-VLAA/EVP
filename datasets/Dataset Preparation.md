# Dataset Preparation



Thank to Pytorch, we can use most datasets by:

```bash 
train_set = torchvision.datasets.cifar100(root, Train = True, transforms = transforms)
val_set = torchvision.datasets.cifar100(root, Train = False, transforms = transforms_test)

```



There are some datasets need special care:

**DMLab**:

DMLab is the dataset from tensorflow datasets, so you can import it from tensorflow datasets, and then transform it to pytorch version. 

```bash 
# DMLab 
ds_train = tfds.load(name="dmlab", split="train", as_supervised=True)
ds_test = tfds.load(name="dmlab", split="test", as_supervised=True)

```



**Resisc45**:

You can download Resisc45 dataset here: http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html



**CLEVR**:

You can download CLEVR dataset here: https://cs.stanford.edu/people/jcjohns/clevr/



Also, you can also prepare above datasets from tensorflow datasets.
