import oss2

def oss_upload(local_path, file_name):
    access_key_id = ''
    access_key_secret = ''

    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket_name = ""
    endpoint = "https://oss-cn-beijing.aliyuncs.com"
    region = "cn-beijing"

    bucket = oss2.Bucket(auth, endpoint, "", region=region)
    file_name = file_name.replace(" ", "")
    object_name = f'agriculture/{file_name}'
    with open(local_path, 'rb') as fileobj:
        bucket.put_object(f'agriculture/{file_name}', fileobj)
    url = f"https://{bucket_name}.oss-cn-beijing.aliyuncs.com/{object_name}"
    return url

if __name__ == '__main__':
    input_path = ".jpg"
    oss_upload(input_path, "test2.jpg")
