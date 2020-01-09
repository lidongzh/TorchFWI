#!/bin/bash
if [ "$HOSTNAME" = "Dolores" ]; then
  echo $HOSTNAME
  rm -rf ./build_dolores
  mkdir build_dolores
  cd build_dolores
  cmake ..
  make -j
  cd ../
else
  rm -rf ./build
  mkdir build
  # cd build
  # cmake ..
  make -j
  cd ../
fi