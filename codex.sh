export no_proxy=""
export http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
export https_proxy="http://sys-proxy-rd-relay.byted.org:8118"

curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm --version
nvm install 20
npm install -g @openai/codex@latest
npm set registry https://bnpm.byted.org
# npx @byted/codex-bridge@latest --model glm-4.7 -y
# codex --sandbox danger-full-access