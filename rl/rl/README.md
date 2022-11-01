# agent

電気系後期実験用の単純な agent です。
自由課題で自分でagentを作りたい人のためのサンプルコードとなっています。

## ファイル構造

* ファイル構造は以下のようになります。

```sh
  agents/
   __init__.py
   foo_agent.py
   bar_agent.py
```

## agentを追加するには

* 他のagentファイルと同様に必要なインタフェース関数を実装し、```agents/__init__.py```内でimportしてください。

* importの詳細については、```agents/__init__.py```に書かれているコメントを参照してください。

# gym_easymaze

電気系後期実験用の単純な迷路 environment です。
自由課題で自分で環境作りたい人のためのサンプルコードとなっています。

[環境の作り方は主にここを参照しています。](https://github.com/openai/gym/blob/master/docs/creating-environments.md "環境の作り方")

## ファイル構造

* ファイル構造は以下のようになります。

```sh
 README.md
 gym_foo/
  __init__.py
  envs/
   __init__.py
   foo_env.py
   foo_hard_env.py
   foo_veryhard_env.py
```

* それぞれの書き方は上記リンク等を参照してください。

## 環境を追加するには

* easymazeの別versionを作る場合、```gym_easymaze/envs/```以下に新たにファイルを作成し、```gym_easymaze/__init__.py```に```register```項目を追加してください。

* 全く別の環境を作る場合は、```gym_easymaze```フォルダごとまるまるコピーしてください。
