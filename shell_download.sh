while IFS='' read -r line || [[ -n "$line" ]]; do
    wget $line	
done < "$1"
