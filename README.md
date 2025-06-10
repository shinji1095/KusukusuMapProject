# KusukusuMapProject

くすくすマップ九州プロジェクトを管理するためのリポジトリです。

# 開発環境

| 項目         | 使用技術・サービス                        | 目的・補足説明                               |
| ---------- | -------------------------------- | ------------------------------------- |
| コンテナ     | Docker                           | アプリケーションの環境構築と本番用コンテナイメージの作成          |
| Webフレームワーク | Django                           | PythonベースのサーバーサイドフレームワークでWeb APIを構築   |
| CI/CD      | Cloud Build                      | Dockerイメージのビルドと Cloud Run への自動デプロイ    |
| アプリ実行環境    | Cloud Run                        | スケーラブルなサーバーレス実行環境。未認証アクセスを許可設定可能      |

# ルート取得

以下のフォーマットのURLでREST APIにアクセスする．

```shell
https://kusukusumapproject-46431021282.asia-northeast2.run.app/route/crosswalk/?origin={lat,lon}&destination={lat,lon}
```

例えば，以下のようにURLを作成してください．

```shell
https://kusukusumapproject-46431021282.asia-northeast2.run.app/route/crosswalk/?origin=33.888729722026795,130.7108332758082&destination=33.891079,130.703475
```

# Contact

| 名前   | メールアドレス                   |
|------|--------------------------------|
| 江藤真士 | eto.shinji786@mail.kyutech.jp |
| 出村翼 | kusukusu.q.fk@gmail.com      |