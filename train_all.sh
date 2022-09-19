
for name in 'gpinn' 'pinn'
  do
    for Nx in 5 10 15 20 25 30
      do
        python run_3.3.1.py --Nx_EQs ${Nx} --net_type ${name}
      done
      echo ${name} + "3.3.1 completed!"
  done


  for name in 'pinn' 'gpinn'
  do
    for Nx in 5 10 15 20 25 30
      do
        python run_3.3.2.py --Nx_EQs ${Nx} --net_type ${name}
      done
      echo ${name} + "3.3.2 completed!"
  done