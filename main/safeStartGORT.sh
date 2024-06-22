#!/bin/bash

ps -lA | grep -i "person_finder.py" | grep -v grep &> /dev/null
retcode=$?

if [ "$retcode" != "0" ]; then # start GORT if not already running.
	echo "Starting GORT"
	/Users/streamer/Documents/Cam5autostart.sh >>/dev/null 2>&1 & 
else 
	echo "GORT already running"
	ps -lA | grep -i "person_finder.py" | grep -v grep
fi

