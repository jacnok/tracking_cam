#!/bin/bash

ps -lA | grep -i "person_finder.py" | grep grep &> /dev/null
retcode=$?

if [ "$retcode" == "0" ]; then # Kill Gort If running
	pid=$(ps -lA | grep -i "person_finder.py" | grep -v grep | awk '{print $2;}')
	echo "Killing pid $pid"
	kill -9 $pid
fi

