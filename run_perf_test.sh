#!/bin/bash

version=$1

echo Preparing to test $version

if [ ! -d output/profiling ]; then
  mkdir -p output/profiling
fi

echo Building...

cmake --build build/$version --parallel -t performance

echo Testing...
echo

git_hash=$(git log --oneline | head -n 1 | awk '{ print $1 }')

for sim_id in {1..14}
do
  for iter in {1..3}
  do
    file=output/profiling/times_${git_hash}_${version}_sim-${sim_id}.${iter}.csv

    echo -e "scene count,$version simulation time,$version total time" > $file
    echo -e "scene count\t$version simulation time\t$version total time"

    for sim_count in {10..82}
      do
        sim_count=$(python3 -c 'print(int(2**('$sim_count'/5)))')
        ./build/$version/tests/performance -m $sim_id -s $sim_count \
          | grep "Kernel took:" \
          | awk '{printf("'$sim_count',%s,%s\n",$4,$7)}' \
          | tee -a $file | awk -F ',' '{print $1,"\t",$2,"\t",$3}' \
          || (echo "Error! $sim_count"; exit 1)
      done

    # run scene 7 with variations in the scenes
    if [ $sim_id == 7 ]
    then
      file=output/profiling/times_${git_hash}_${version}_sim-${sim_id}_vars.${iter}.csv
      echo -e "scene count,$version simulation time,$version total time" > $file
      echo -e "scene count\t$version simulation time\t$version total time"
      for sim_count in {10..82}
        do
          sim_count=$(python3 -c 'print(int(2**('$sim_count'/5)))')
          ./build/$version/tests/performance -m $sim_id -s $sim_count -v \
            | grep "Kernel took:" \
            | awk '{printf("'$sim_count',%s,%s\n",$4,$7)}' \
            | tee -a $file | awk -F ',' '{print $1,"\t",$2,"\t",$3}' \
            || (echo "Error! $sim_count"; exit 1)
        done
    fi
  done

  cd output/profiling 

  python3 data_processing.py ${git_hash} ${version} sim-${sim_id}

  if [ $sim_id == 7 ]
  then
    python3 data_processing.py ${git_hash} ${version} sim-${sim_id}_vars
  fi

  cd ../..
done

echo
echo Done
