#!/usr/bin/bash

# Author: Lajos Palanki [LP618]
# E-mail: LP618@ic.ac.uk
# Version: 1.4

usage="$(basename "$0") [-h] [-s n] [-d detector] [-z n] [-S n] [-E n] [-O file] [-f n] [-m TRUE|FALSE] [-v] [-F]

This program automates and makes several things about data acquisition easier.
It initially resets the position to the desired start and then takes data according to the options.
Its output is two files: the default .txt file renamed according to the -O option and a csv with a first line header with information on the data acquisition

where:
	-h  show this help text
	-s  set the speed value [default: 0.002]
	-d  set detector [default:/dev/ttyUSB0]
	-z  set zero point [default:0]
	-S  set start point [default:Current position]
	-m  if set to true and start position not set, speed is ignored and no data will be taken. [default:TRUE]
	-E  set end point [default:10]
	-O  set output file name without extension[default:./Output_data]
	-f  Sampling frequency [default:50] [!NOT RECOMMENDED TO CHANGE!]
	-v  Verbose. Prints progress
	-F  Forces override of files

Script written by: Lajos Palanki
E-mail: LP618@ic.ac.uk
Last Edit:2020/03/03"

trap ctrl_c 2

ctrl_c(){

	echo -e "" >&3
	echo -e "$detector,$START,$END,$speed,$Frequency,$(python -c "print($speed / $Frequency)"),$ZERO_POINT" #Metadata
	head -n-1 Output_data.txt
	echo "Data taking interrupted at $(tail -n 2 ./log.txt | head -n 1 | awk '{print $5}') mm" >&3
	echo "Renaming output to $(python -c "print('$Output'+'.txt')")" >&3
	mv Output_data.txt $(python -c "print('$Output'+'.txt')")
	echo "Complete" >&3
	kill %1
	exit 1

}

monitor(){

	cols=$( tput cols )
	rows=$( tput lines )
	sleep 2
	echo -e "|Current position\t|Completion\t|Remaining time\t|"
	echo -e ""
	echo -e ""
	while :
	do
		if [[ "$(tail -n 1 ./log.txt)" == *"*"* ]] || [[ "$(tail -n 1 ./log.txt)" == *"Error"* ]] # Possible source of error
		then
			break
		fi
		tput cup $(($rows-3)) 0
		tput ed
		progress=$(tail -n 2 ./log.txt | head -n 1 | awk '{print $5}')
		perc=$(python -c "print(100*($progress-$1)/($2 - $1))")
		tput cup $(($rows-3)) 0
		echo -e "|$progress\t\t|$(python -c "print(str(int($perc))+'%')")\t|$(python -c "print(int(($2-$progress)/($3)))")\t|"
		tput cup $(($rows-2)) 0
		echo -e "["
		tput cup $(($rows-2)) 1
		echo -e "$(python -c "print('#'*int(($cols-2)*$perc/100))")"
		tput cup $(($rows-2)) $(($cols-1))
		echo -e "]"
		sleep 1
	done
}


speed=0.002
detector=/dev/ttyUSB0
ZERO_POINT=0
START=Curr_pos
END=Unset
Output=./Output_data
Frequency=50
move=TRUE
verbose=FALSE
force=FALSE
while getopts ':h:s:d:z:S:E:O:f:m:vF' option
do
  case "$option" in
    h)
       echo "$usage"
       exit
       ;;
    s)
       speed=$OPTARG
       ;;
    d)
       detector=$OPTARG
       ;;
    z)
       ZERO_POINT=$OPTARG
       ;;
    S)
       START=$OPTARG
       ;;
    E)
       END=$OPTARG
       ;;
    O)
       Output=$OPTARG
       ;;
    f)
       Frequency=$OARGPT
       ;;
    m)
       move=$OPTARG
       ;;
    v)
       verbose=TRUE
       ;;
    F)
       force=TRUE
       ;;
    :)
       printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?)
       printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

tabs 16
exec 3>&2

if [ ! -e $detector ]
then
	echo -e "It appears $detector does not exist on this system. Make sure detector is connected" >&3
	echo -e "Possible that the connected detector is somewhere else. Possible candidates:" >&3
	echo $(ls /dev/ | grep -i USB) >&3
	exit 1
fi

if [ -a $(python -c "print('$Output'+'.csv')") ] && [ "$force" == "FALSE" ] && [ "$Output" != "./Output_data" ]
then
	echo -e "$(python -c "print('$Output'+'.csv')") already exsists. Use an other output or the -F argument." >&3
	exit 1
fi

exec 4>./log.txt # Log file. Can be used less on to see progress Output of take-data redirected
exec 1>$(python -c "print('$Output'+'.csv')")

if [ "$END" = "Unset" ]
then
	echo -e "End position is unset. Setting end position to default of 10mm" >&3
	END=10
fi

if [ "$START" == "Curr_pos" ]
then
    if [ "$move" == "TRUE" ]
    then
        echo -e "No starting position specified, moving from current position to $END. No data will be taken. Speed is ignored and set to 1" >&3
        take-data $END 10 $detector 1 >&4
        exit 1
    else
        echo -e "Attempting to find current Position. This might take a second." >&3 #For possible location.
        take-data 0 0.0001 $detector 1 >&4
        sleep 2
        start_pos=$(cat Output_data.txt | awk '{print $5}')
        echo -e "Identified Starting position as $start_pos" >&3
        START=$start_pos
    fi
fi

echo -e "Starting Scan reset"  >&3
take-data $START 10 $detector 10 >&4
#rm Output_data.txt # Removed due to issues and lack of function
sleep 1
echo -e "Scan reset complete" >&3

echo -e "Starting Scan" >&3
echo -e "|Detector\t|From\t|To\t|Speed\t|Sampling\t|Run-time\t|" >&3
echo -e "|$detector\t|$START\t|$END\t|$speed\t|$Frequency\t|$(python -c "print(int(($END - $START) / $speed))")\t|" >&3

if [ "$verbose" == "TRUE" ]
then
	monitor $START $END $speed <&0 >&3 &
	monitor_PID=$!
fi
take-data $END $speed $detector $(python -c "print(1+ ($Frequency * ($END - $START)/ $speed))") >&4



echo -e "$detector,$START,$END,$speed,$Frequency,$(python -c "print($speed / $Frequency)"),$ZERO_POINT" #Metadata
cat Output_data.txt

echo "Data taking finished. Results:" >&3
tail -n 4 ./log.txt >&3
echo "Renaming output to $(python -c "print('$Output'+'.txt')")" >&3
mv Output_data.txt $(python -c "print('$Output'+'.txt')")
echo "Complete" >&3
