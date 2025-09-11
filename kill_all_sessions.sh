#!/bin/bash

# Script to kill all Gaussian Splatting training sessions

SESSION_NAME="gaussian_training"

echo "Killing all Gaussian Splatting training sessions..."

for gpu in {0..3}; do
    session_name="${SESSION_NAME}_gpu${gpu}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Killing session: $session_name"
        tmux kill-session -t "$session_name"
    else
        echo "Session $session_name not found"
    fi
done

echo "All sessions killed."
echo ""
echo "To check remaining sessions: tmux list-sessions"
