import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ..content import ORDER, BUDGETS
from ..server import Application, handler


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.app=Application(self.temp.name)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),handler(self.app))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.token=self.app.store.create(count=1,order='progressive')[0]['token']

    def tearDown(self):
        self.app.stopping.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.app.runner.pool.shutdown()
        self.temp.cleanup()

    def call(self,path,data=None,headers=None):
        headers=headers or {}
        if data is not None:
            headers={'Content-Type':'application/json',**headers}
        request=Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
        try:
            response=urlopen(request,timeout=5)
        except HTTPError as exc:
            response=exc
        with response:
            return response.status,json.loads(response.read())

    def test_admin_origin_and_hidden_grade_controls(self):
        self.assertEqual(self.call('/api/admin')[0],401)
        path=f'/api/p/{self.token}/action'
        self.assertEqual(self.call(path,{'action':'consent','agree':True},{'Origin':'https://elsewhere.invalid'})[0],403)
        self.assertEqual(self.app.store.state(self.token)['stage'],'consent')
        self.assertEqual(self.call(path,{'action':'locale','locale':'en','condition':'rust'})[0],403)
        self.assertEqual(self.call(f'/api/p/{self.token}/run',{'purpose':'grade'})[0],400)
        self.assertEqual(self.call('/api/admin/create',{'count':1})[0],401)

    def test_connected_timer_freezes_and_schedules_latest_draft(self):
        now=[1000.]
        self.app.store.clock=lambda:now[0]
        store=self.app.store
        store.action(self.token,'consent',{'agree':True})
        store.action(self.token,'background',{'answers':dict.fromkeys(['python','rust','arrays','zk'],'basic')})
        store.action(self.token,'tutorial_done',{})
        run=store.new_run(self.token,'examples')
        store.update_run(run['id'],'complete',{'status':'passed'})
        store.action(self.token,'practice_done',{})
        store.action(self.token,'start_task',{})
        store.save(self.token,'latest frozen source',0,ORDER[0])
        observed=[]
        scheduled=threading.Event()
        def schedule(_store,job):
            if job:
                observed.append(job)
                scheduled.set()
        self.app.runner.schedule=schedule
        path=f'/api/p/{self.token}/presence'
        self.assertEqual(self.call(path, {'action':'open','page_id':'api-test-page','task':ORDER[0]})[0],200)
        for _ in range(BUDGETS[ORDER[0]]):
            now[0]+=1
            store.presence(self.token,'heartbeat','api-test-page',ORDER[0])
        self.app.start_clock()
        self.assertTrue(scheduled.wait(3),'timer should grade an active task at its deadline')
        self.assertEqual(observed[0]['code'],'latest frozen source')
        self.assertEqual(observed[0]['purpose'],'grade')
        self.assertEqual(store.state(self.token)['stage'],'ready')

    def test_attempt_trail_is_private_and_scheduled_after_session(self):
        store=self.app.store
        with store.db() as db:
            db.execute("UPDATE participants SET stage='task' WHERE token=?",(self.token,))
            store.start_session(db,store.participant(db,self.token),ORDER[0])
        seen=[];done=threading.Event()
        def schedule(_store,job,cases=None):
            if not job:return
            seen.append(job)
            if job['purpose']=='attempt_grade':
                _store.update_run(job['id'],'complete',{'status':'failed','cases':[{'passed':False,'input':'PRIVATE_INPUT','expected':'PRIVATE_OUTPUT'}]})
                if len([r for r in seen if r['purpose']=='attempt_grade'])==2:done.set()
            else:_store.update_run(job['id'],'complete',{'status':'compile_error'})
        self.app.runner.schedule=schedule
        path=f'/api/p/{self.token}'
        for rev,code in [(0,'attempt one'),(1,'attempt two')]:
            status,result=self.call(path+'/run',{'purpose':'examples','code':code,'revision':rev,'task':ORDER[0]})
            self.assertEqual(status,202,result)
        self.app.start_clock()
        self.assertFalse(done.wait(.1))
        data=store.export(token=self.token)
        hidden=[r for r in data['runs'] if r['purpose']=='attempt_grade']
        self.assertEqual(len(hidden),2)
        for run in hidden:self.assertEqual(self.call(path+'/runs/'+run['id'])[0],404)
        self.assertEqual(self.call(path+'/run',{'purpose':'attempt_grade'})[0],400)
        self.assertEqual(self.call('/api/admin/participant/'+self.token)[0],401)
        self.assertNotIn('attempt_grade',json.dumps(self.call(path+'/state')[1]))
        self.call(path+'/action',{'action':'withdraw'})
        self.assertTrue(done.wait(3))
        self.assertEqual([r['code'] for r in seen if r['purpose']=='attempt_grade'],['attempt one','attempt two'])
        for run in hidden:self.assertEqual(self.call(path+'/runs/'+run['id'])[0],404)
        headers={'Authorization':'Bearer '+store.config['admin_key']}
        status,data=self.call('/api/admin/participant/'+self.token,headers=headers)
        self.assertEqual(status,200)
        self.assertEqual(len(data['attempt_trail']),2)
        self.assertTrue(all(r['hidden_status']=='failed' for r in data['attempt_trail']))
