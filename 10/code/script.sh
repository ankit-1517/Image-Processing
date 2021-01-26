# 1st_manual/21_manual1.gif, images/21_training.tif
datasetPath=/home/ankit/Desktop/Acad/Sem/Sem6/ELL715/Assn/10/datasets/training
# groundtruth/21_manual.gif, origin/21_training.tif
trainPath=/home/ankit/Desktop/Acad/Sem/Sem6/ELL715/Assn/10/Retina-VesselNet/experiments/VesselNet/dataset/train
# groundtruh, origin format same as trainPath
validatePath=/home/ankit/Desktop/Acad/Sem/Sem6/ELL715/Assn/10/Retina-VesselNet/experiments/VesselNet/dataset/validate
# groundtruth, origin format same as trainPath
testPath=/home/ankit/Desktop/Acad/Sem/Sem6/ELL715/Assn/10/Retina-VesselNet/experiments/VesselNet/test
resultPath=/home/ankit/Desktop/Acad/Sem/Sem6/ELL715/Assn/10/Retina-VesselNet/results

testArr=( 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 )
temp1=0
temp2=0

for i in "${!testArr[@]}"; do 
	echo ${testArr[i]};
	temp1=$(expr $i - 1)
	if (( $temp1 < 0 )); then
		temp1=$(expr $temp1 + 20)
	fi
	temp2=$(expr $i - 2)
	if (( $temp2 < 0 )); then
		temp2=$(expr $temp2 + 20)
	fi
	for j in "${!testArr[@]}"; do 
		# test
		if [ $i = $j ]; then
			cp "$datasetPath/1st_manual/"${testArr[j]}"_manual1.gif" "$testPath/groundtruth/"${testArr[j]}"_manual1.gif"
			cp "$datasetPath/images/"${testArr[j]}"_training.tif" "$testPath/origin/"${testArr[j]}"_training.tif"
		else
			# validation
			if [[ $j = $temp1 || $j = $temp2 ]]; then
				cp "$datasetPath/1st_manual/"${testArr[j]}"_manual1.gif" "$validatePath/groundtruth/"${testArr[j]}"_manual1.gif"
				cp "$datasetPath/images/"${testArr[j]}"_training.tif" "$validatePath/origin/"${testArr[j]}"_training.tif"
			# training
			else
				cp "$datasetPath/1st_manual/"${testArr[j]}"_manual1.gif" "$trainPath/groundtruth/"${testArr[j]}"_manual1.gif"
				cp "$datasetPath/images/"${testArr[j]}"_training.tif" "$trainPath/origin/"${testArr[j]}"_training.tif"
			fi
		fi
	done
	wait
	mkdir $resultPath/${testArr[i]}
	stdbuf -oL python3 main_train.py > "$resultPath/"${testArr[i]}"/train.txt"
	wait
	stdbuf -oL python3 main_test.py > "$resultPath/"${testArr[i]}"/test.txt"
	wait
	cp "$testPath/result/_merge.jpg" "$resultPath/"${testArr[i]}"/out.jpg"
	cp "$resultPath/DRIVE_ROC.png" "$resultPath/"${testArr[i]}"/plot.png"
	# remove all images
	# train
	rm -r $trainPath/groundtruth/*
	rm -r $trainPath/origin/*
	# validate
	rm -r $validatePath/groundtruth/*
	rm -r $validatePath/origin/*
	# test
	rm -r $testPath/groundtruth/*
	rm -r $testPath/origin/*
	rm -r $testPath/result/*
	rm -r $testPath/result/*
	wait
	break
done

