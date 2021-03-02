py-wsi-fork
===========

This is a fork of py-wsi (https://github.com/ysbecca/py-wsi). It contains the following additional functionality:

-  Conversion of segmentation ground truth (gt) provided in xml format to a region matching a certain DeepZoom level
-  Optionally storing the gt region together with the respective image patch in an LMDB item or on disk (previously it was only possible to assign a single label to each patch and not to store segmentation gt)
-  Functionality to display a WSI slide partitioned into patches together with its gt annotation
-  Bugfix related to a problem occurring when storing overlapping patches in the databases

