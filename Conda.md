---


---

<h1 id="basic-conda">Basic conda</h1>
<h2 id="create-activate-and-delete-an-environment">Create, activate and delete an environment</h2>
<ol>
<li>
<p>create a new environment choosing the name (you can also specify the python version and also specific packages you need, as rasterio in the example belo)</p>
<pre><code>conda create  myenv python=3.10 rasterio 
</code></pre>
</li>
<li>
<p>inspect environment list</p>
<pre><code>conda env list
</code></pre>
</li>
<li>
<p>activate/deactivate environment</p>
<pre><code>conda activate myenv
conda deactivate
</code></pre>
</li>
<li>
<p>delete</p>
<pre><code>conda remove myenv --all
</code></pre>
</li>
</ol>
<h2 id="install-packages">Install packages</h2>
<p>There are different commands to install a packages, accordingly to the source.</p>
<p>Indeed, there are also package that are not available through the <a href="https://anaconda.org/anaconda/repo">official anaconda repository</a> or <a href="https://conda-forge.org/packages/">conda forge</a>. The last is a community-maintained collection of conda packages separated from the default Anaconda one. However a lot of package are also available on GitHub.</p>
<ol>
<li>anaconda repository<br>
<code>pip install PACKAGE_NAME</code></li>
<li>conda forge<br>
<code>conda install -c conda-forge PACKAGE_NAME</code></li>
<li>GitHub<br>
<code>pip install git+https://github.com/username/repo.git</code></li>
</ol>
<p>From GitHub we can also <strong>copy an entire repository</strong></p>
<pre><code>git clone https://github.com/username/repo.git
</code></pre>
<h2 id="launch-a-.py-script">Launch a .py script</h2>
<p>Select the proper folder where the .py and other utils needed are stored.</p>
<h3 id="select-directory">1. select directory</h3>
<p>On windows</p>
<pre><code>cd "directory/path"
</code></pre>
<p>If we use an external memory</p>
<p><code>G:</code> or <code>D:</code> accordingly to what is the name of my SSD</p>
<pre><code>cd FU\Anaconda\SEN2\SNAP_GraphBuilder 
</code></pre>
<p>or directly</p>
<pre><code>cd "G:\FU\Anaconda\SEN2\Sentinel2_mosaicking_1image"
</code></pre>
<h3 id="launch-.py">launch .py</h3>
<pre><code>python Sentinel2_Multibands_mosaick.py
</code></pre>
<p>If we want to run it while we are using python environment</p>
<pre><code>python
