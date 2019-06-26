Person re-identification based on deep metric learning 
=============
We compared the performance of deep metric losses commonly used in recent years on person re-identification task. 

Features
-------
- no tricks
    - no Bottleneck after backbone
    - no stride=2-->1 in last conv layer
	- no warm-up strategy 
	- no label smmothing when applying cross entropy loss 
- few data augmentation
    - only resize/normalization/randomhorizontalflip when training
    - no flipping when testing
- a dozen of deep metric losses

The backbone is resnet-50 and datasets are Market1501 and DukeMTMC-reid. 

The program is based on [torchreid](https://github.com/KaiyangZhou/deep-person-reid) framework by [KaiyangZhou](https://github.com/KaiyangZhou).

Losses
------
- Softmax
- Triplet loss
- Softmax + Triplet loss
- Center loss + Softmax
- N-pair loss
- Histogram loss
- Focal loss
- Multi-similarity loss
- A-softmax loss(Sphereface, no margin)
- A-softmax loss(Sphereface, margin=4)
- LMCL loss(Cosface)
- Additive angular margin loss(Arcface)

Results
--------
<table>
    <tr>
        <th rowspan="2">Loss</th>
        <th colspan="2">Market-1501</th>
        <th colspan="2">DukeMTMC-reid</th>
    </tr>
    <tr>
        <td>rank-1</td>
        <td>mAP</td>
         <td>rank-1</td>
        <td>mAP</td>
    </tr>
    <tr>
        <td>Softmax</td>
        <td>82.3%</td>
        <td>64.6%</td>
        <td>72.6%</td>
        <td>54.9%</td>
    </tr>
    <tr>
        <td>Triplet loss</td>
        <td>77.9%</td>
        <td>59.2%</td>
        <td>70.4%</td>
        <td>52.6%</td>
    </tr>
    <tr>
        <td>Softmax+Triplet loss</td>
        <td>83.7%</td>
        <td>66.8%</td>
        <td>75.1%</td>
        <td>57.3%</td>
    </tr>
    <tr>
        <td>Center loss+Softmax</td>
        <td>84.2%</td>
        <td>66.1%</td>
        <td>73.3%</td>
        <td>55.9%</td>
    </tr>
    <tr>
        <td>N-pair loss</td>
        <td>79.8%</td>
        <td>62.2%</td>
        <td>75.0%</td>
        <td>57.2%</td>
    </tr>
    <tr>
        <td>Histogram loss</td>
        <td>75.9%</td>
        <td>59.6%</td>
        <td>68.3%</td>
        <td>50.1%</td>
    </tr>
    <tr>
        <td>Focal loss</td>
        <td>82.8%</td>
        <td>64.2%</td>
        <td>72.8%</td>
        <td>54.1%</td>
    </tr>
    <tr>
        <td>Multi-similarity loss</td>
        <td>72.8%</td>
        <td>55.9%</td>
        <td>66.0%</td>
        <td>47.3%</td>
    </tr>
    <tr>
        <td>Sphereface (no margin)</td>
        <td>85.6%</td>
        <td>69.2%</td>
        <td>75.6%</td>
        <td>58.1%</td>
    </tr>
    <tr>
        <td>Sphereface (margin=4)</td>
        <td>86.2%</td>
        <td>70.0%</td>
        <td>76.0%</td>
        <td>58.1%</td>
    </tr>
    <tr>
        <td>Cosface</td>
        <td>84.9%</td>
        <td>65.1%</td>
        <td>75.0%</td>
        <td>54.7%</td>
    </tr>
    <tr>
        <td>Arcface</td>
        <td>85.1%</td>
        <td>65.0%</td>
        <td>75.2%</td>
        <td>54.3%</td>
    </tr>
</table>

Reproduction
--------
1. Install the framework and create the environment
https://github.com/KaiyangZhou/deep-person-reid#installation
2. Prepare the datasets
https://kaiyangzhou.github.io/deep-person-reid/datasets.html#image-datasets
3. Run `run.sh`


To do
------
- [x] add Cosface and Arcface
- [x] update README.md
- [ ] try more datasets
