---
title:  "How to use Jupyter Notebook with Spark on Ubuntu"
date:   2017-05-11
tags: [Miscellaneous]

excerpt: "Spark, jupyter notebook, and ubuntu"
---


In this post, I have outlined the protocol that I used to enable Jupyter notebooks to run Spark on my Ubuntu computer. 
I have followed, with a few tweaks, the instructions in a blog [post](!http://blog.thedataincubator.com/2017/04/spark-2-0-on-jupyter-with-toree/) by [The Data Incubator](!https://www.thedataincubator.com).
I have the [Python 2.7 version of Anaconda](!https://www.continuum.io/downloads), which comes with Jupyter pre-installed.

## Install Java
<div class="box">
  <pre>$sudo apt-get install default-jre
$sudo apt-get install default-jdk</pre>
</div>
Java installation can be verified using 
<div class="box">
  <pre>$java -version</pre>
</div>
I get the following response from this command
<div class="box">
  <pre>openjdk version "1.8.0_131"
OpenJDK Runtime Environment (build 1.8.0_131-8u131-b11-0ubuntu1.16.04.2-b11)
OpenJDK 64-Bit Server VM (build 25.131-b11, mixed mode)</pre>
</div>

## Install Spark
Download Spark from http://spark.apache.org/downloads.html and simply extract the contents of the .tgz file as follows.
<div class="box">
  <pre>$tar zxvf spark-2.0.1-bin-hadoop2.4.tgz</pre>
</div>

## Setup paths 
Add the following two lines to shell's startup script, ~/.bashrc in my case. I installed Spark in the directory /usr/local/share/spark/.
<div class="box">
  <pre>export SPARK_HOME=/usr/local/share/spark/spark-2.0.1-bin-hadoop2.4 
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$SPARK_HOME/python/lib</pre>
</div>

Now run the '.bashrc'
<div class="box">
  <pre>$source ~/.bashrc</pre>
</div>

## Install Apache Toree
Install Toree and configure Jupyter to run Toree as follows.
<div class="box">
  <pre>$pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.2.0/snapshots/dev1/toree-pip/toree-0.2.0.dev1.tar.gz
$jupyter toree install --user</pre>
</div>

## Install py4j
<div class="box">
  <pre>$pip install py4j</pre>
</div>

That's it! Now we can use PySpark from Jupyter Notebooks.


```python
from pyspark import SparkContext
sc = SparkContext("local[*]", "temp")
print sc.version
```

    2.0.1

