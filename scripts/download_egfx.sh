#/bin/bash
mkdir -p data
cd data
mkdir -p egfx
cd egfx
wget https://zenodo.org/record/7044411/files/BluesDriver.zip?download=1 -O BluesDriver.zip
wget https://zenodo.org/record/7044411/files/Chorus.zip?download=1 -O Chorus.zip
wget https://zenodo.org/record/7044411/files/Clean.zip?download=1 -O Clean.zip
wget https://zenodo.org/record/7044411/files/Digital-Delay.zip?download=1 -O Digital-Delay.zip
wget https://zenodo.org/record/7044411/files/Flanger.zip?download=1 -O Flanger.zip
wget https://zenodo.org/record/7044411/files/Hall-Reverb.zip?download=1 -O Hall-Reverb.zip
wget https://zenodo.org/record/7044411/files/Phaser.zip?download=1 -O Phaser.zip
wget https://zenodo.org/record/7044411/files/Plate-Reverb.zip?download=1 -O Plate-Reverb.zip
wget https://zenodo.org/record/7044411/files/RAT.zip?download=1 -O RAT.zip
wget https://zenodo.org/record/7044411/files/Spring-Reverb.zip?download=1 -O Spring-Reverb.zip
wget https://zenodo.org/record/7044411/files/Sweep-Echo.zip?download=1 -O Sweep-Echo.zip
wget https://zenodo.org/record/7044411/files/TapeEcho.zip?download=1 -O TapeEcho.zip
wget https://zenodo.org/record/7044411/files/TubeScreamer.zip?download=1 -O TubeScreamer.zip
unzip -n \*.zip
rm -rf *.zip


