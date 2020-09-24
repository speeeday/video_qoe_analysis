from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class Drive_Upload_Wrapper():
	def __init__(self):
		g_login = GoogleAuth(settings_file='metadata/pydrive_settings.yaml')
		g_login.LocalWebserverAuth()
		self.drive = GoogleDrive(g_login)

	def upload_file(self, fpath):
		with open(fpath, 'r') as f:
			file_drive = self.drive.CreateFile({
					"title": f.name
				})
			file_drive.SetContentString(f.read())
			file_drive.Upload()

	def upload_pickle(self, pickpath):
		new_drive_file = self.drive.CreateFile()
		new_drive_file.SetContentFile(pickpath)
		new_drive_file.Upload()

if __name__ == "__main__":
	from subprocess import call
	import pickle
	x = {'foo': 4}
	pickle.dump(x, open('tst.pkl','wb'))
	duw = Drive_Upload_Wrapper()
	duw.upload_pickle('tst.pkl')

	call('rm tst.pkl', shell=True)
