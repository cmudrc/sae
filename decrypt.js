var keySize = 256;
var iterations = 1000;
function decrypt (encryptedMsg, pass) {
    var salt = CryptoJS.enc.Hex.parse(encryptedMsg.substr(0, 32));
    var iv = CryptoJS.enc.Hex.parse(encryptedMsg.substr(32, 32))
    var encrypted = encryptedMsg.substring(64);

    var key = CryptoJS.PBKDF2(pass, salt, {
        keySize: keySize/32,
        iterations: iterations
    });

    var decrypted = CryptoJS.AES.decrypt(encrypted, key, {
        iv: iv,
        padding: CryptoJS.pad.Pkcs7,
        mode: CryptoJS.mode.CBC
    }).toString(CryptoJS.enc.Utf8);
    return decrypted;
}

document.getElementById('staticrypt-form').addEventListener('submit', function(e) {
    e.preventDefault();

    var passphrase = document.getElementById('staticrypt-password').value,
        encryptedHMAC = encryptedMsg.substring(0, 64),
        encryptedHTML = encryptedMsg.substring(64),
        decryptedHMAC = CryptoJS.HmacSHA256(encryptedHTML, CryptoJS.SHA256(passphrase).toString()).toString();

    if (decryptedHMAC !== encryptedHMAC) {
        alert('Password incorrect');
        return;
    }

    var plainHTML = decrypt(encryptedHTML, passphrase);

    document.write(plainHTML);
    document.close();
});
