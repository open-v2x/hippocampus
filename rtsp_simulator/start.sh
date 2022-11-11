#!/bin/bash

set -e

if [ "$SOURCE_URL" != "" ]; then
   echo "Starting rtsp server..."
   rtsp-simple-server rtsp-simple-server.yml &
   sleep 2

   echo "Start relaying from $SOURCE_URL to rtsp://0.0.0.0:8554/$STREAM_NAME"
   while true; do
      set -x
      ffmpeg $FFMPEG_INPUT_ARGS -re -stream_loop -1 -i $SOURCE_URL $FFMPEG_OUTPUT_ARGS $RTSP_PROXY_SOURCE_TCP -preset ultrafast -f rtsp rtsp://localhost:8554/$STREAM_NAME
      set +x
      echo "Reconnecting..."
      sleep 1
   done
else
   echo "Won't restream a source feed to the server because SOURCE_URL was not defined"
   echo "Starting rtsp server. You can publish feeds to it (ex.: ffmpeg -i somesource.mjpg -c copy -f rtsp rtsp://localhost:8554/myfeed)"
   rtsp-simple-server rtsp-simple-server.yml
fi