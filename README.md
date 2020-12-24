![Build](https://github.com/dakatk/rust-nn-cli/workflows/Build/badge.svg?branch=master) ![Clippy](https://github.com/dakatk/rust-nn-cli/workflows/Clippy/badge.svg?branch=master)

# Open Neural Network Benchmarker

This is the Open Neural Network Benchmarker, or ONNB, which is a command-line application written for the sole purpose of testing various neural network setups. The goal of this project is to make Neural Network assessment simple and flexible. 

This tool is the spritual successor to a personal project ~~that is still under active development~~ called Sensei, which is a tool I use for observiing the activity of different neural network setups. Instead of rust, this tool is written in Python, so it's much more flexible and has access to matplotlib. Hoever, it's also significantly slower and doesn't have the optimization or safety features that Rust has. If you'd like to play around with this code, or if you want to get a feel for how this stuff works and you understand Python better than Rust, then here's a link to the Github repo: https://github.com/dakatk/Machine-Learning-Library*

**Note: That project is old and not my best work. It was mostly stitched together from a series of Youtube videos and barebones examples. This project is meant to be a continuation of that project, now that I understand the mathematics a bit better. Of course, this project still only exists for educational purposes, but I'm much happier with the code I've written for ONNB so far than what I wrote for Sensei*

### Some features anticipated for the final product are:
 - Setups for various types of neural networks (FFNNs, CNNs, RNNs)
 - The ability to load many JSON files from separate directories (one directory for network structures, one directory for training data, one directory for testing data) for training in parallel
 - Saving the internal data of each layer to JSON files at user-defined intervals

*Note: If you are considering downloading/using the code for this application, and are currently reading this, then that means the project is still in it's infancy and is not yet considered stable. At the time of writing, I'm still relatively new to Rust, and this project has only scarcely been tested at it's first milestone. Feel free to use/distribute this code at your own risk*
