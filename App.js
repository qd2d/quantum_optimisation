const cors=require('cors');
const XMLHttpRequest=require('xmlhttprequest').XMLHttpRequest

const corsOptions ={
   origin:'*',
   credentials:true,            //access-control-allow-credentials:true
   optionSuccessStatus:200,
}

const express = require('express')
const {spawn} = require('child_process');
const app = express()

const fs = require('fs')

const key = fs.readFileSync('./CA/localhost/localhost.decrypted.key');
const cert = fs.readFileSync('./CA/localhost/localhost.crt');
const https = require('http');
const server = https.createServer({ key, cert }, app);

const port = process.env.PORT || 8001


const filepath = './data.json'

const allowedOrigins = [
   'http://localhost',
  'http://localhost:8000',
  'https://localhost:8081',
    'https://xxxxx.xxxxx.xx',
    'https://i.ibb.co',
    '*'
];


app.use(cors(corsOptions)) // Use this after the variable declaration */
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header(
    'Access-Control-Allow-Headers',
    'Authorization, Origin, X-Requested-With, Content-Type, Accept'
  );
  res.header("Access-Control-Allow-Methods", "GET, PUT, POST");


  res.header('Access-Control-Allow-Credentials', 'true');

  next();
});
//app.use(express.static(__dirname + '/public')); // exposes index.html, per below
/*
var requestOptions = {
    hostname: 'http://localhost', // url or ip address
    port: 8000, // default to 80 if not provided
    path: '/upload',
    method: 'POST' // HTTP Method

};
var externalRequest = https.request(requestOptions, (externalResponse) => {

    // ServerB done responding
    externalResponse.on('end', () => {

        // Response to client
        res.end('data was send to serverB');

    });
});
*/
let req = new XMLHttpRequest();

req.onreadystatechange = () => {
  if (req.readyState == XMLHttpRequest.DONE) {
    console.log(req.responseText);
  }
};


app.get('/', (req1, res) =>{

var dataToSend;
 // spawn new child process to call the python script
/* const python = spawn('python', ['getimg_url.py']);
 // collect data from script
 python.stdout.on('data', function(data) {

  console.log('Pipe data from python script ...');
  dataToSend = data.toString();
  fs.writeFileSync('data2.json', JSON.stringify(dataToSend));
  console.log(`${data}`);
 });
 // in close event we are sure that stream from child process is closed
 python.on('close', (code) => {
 console.log(`child process close all stdio with code ${code}`);
 // send data to browser
 res.send(JSON.stringify(dataToSend))
 });*/
 // send to external address


// Free to send anything to serverB
// spawn new child process to call the python script
 const python = spawn('python', ['-u','getimg_url.py']);
 // collect data from script


 python.stdout.on('data', function(data) {


     console.log('Pipe data from python script ...');
     dataToSend = data.toString();

  fs.writeFileSync('data.json', JSON.stringify(dataToSend));
  console.log(`${data}`);

 });

 // in close event we are sure that stream from child process is closed
 python.on('close', (code) => {
 console.log(`child process close all stdio with code ${code}`);
 // send data to browser

req.open("PUT", "https://api.jsonbin.io/v3/b/626e0d3538be296761fa7bb7", false);
req.setRequestHeader("Content-Type", "application/json");
req.setRequestHeader("X-Master-Key", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
req.setRequestHeader("secret-key", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
s = dataToSend.replace('\r\n', '');
payload = '{"message":"' + s + '"}';
//req.send('{"sample": "' + dataToSend +'"}')

res.send(JSON.stringify(payload));
req.send(payload);



 });


})
const python2 = spawn('python', ['openurl.py']);

           python2.stderr.on('data', function (data) {
               responseData += data.toString();
           });
           python2.on('close', (code) => {
               console.log(`child process close all stdio with code ${code}`);
           })
 // send data to browser
app.listen(port, () => console.log(`Example app listening on https://localhost: 
${port}!`))//server.
