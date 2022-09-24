#bash
#0.00001, 0.0001, 0.001,
for g in  0.01 0.1
  do
    for name in 'gpinn'
      do
        for Nx in 10 12 14 16 18 20
          do
            python run_3.3.1.py --Nx_EQs ${Nx} --g_weight ${g}  --net_type ${name}+$"$g"
            echo ${name}+$"$g"+"-"+$"$Nx"+" completed!"
          done
      done
  done


