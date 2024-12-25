 Script
mkdir -p .assets/models

if ! command -v aria2c &> /dev/null; then
    echo "aria2c 未安装，正在安装..."
    # 根据操作系统选择合适的安装方式
    if [ -f /etc/debian_version ]; then
        sudo apt-get update && sudo apt-get install -y aria2
    elif [ -f /etc/redhat-release ]; then
        sudo yum install -y epel-release && sudo yum install -y aria2
    else
        echo "未知操作系统，请手动安装 aria2c"
        exit 1
    fi
fi

aria2c -d .assets/models -i url.txt -j 5 --retry-on-download-result=always --max-tries=3 --log=download.log
